import os
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#############################################
# Helper: Convert token string to list of ints
#############################################
def tokenize_output(output_str):
    """
    Given a space-separated string of token indices, return a list of ints.
    """
    tokens = output_str.strip().split()
    return [int(tok) for tok in tokens]

#############################################
# New Dataset: ClevrStepsDatasetNew
# (Loads questions from separated HDF5 file using a fraction of the data)
#############################################
class ClevrStepsDatasetNew(Dataset):
    def __init__(self, annotations_h5_path, image_features_h5_path, data_fraction=1.0):
        """
        Loads annotated questions from a separated HDF5 file (each question stored as a group)
        and the image features.
        
        For spatial outputs, the tokenized "output_values" is assumed to be in groups of 6 tokens:
            [ token, num1, num2, num3, num4, token ]
        We remove the first and last tokens so that the model receives a tensor of shape [T, 4].
        
        For nonspatial outputs, we assume a single token.
        
        data_fraction (a float in (0,1]) specifies the proportion of question groups to load.
        """
        self.samples = []
        # Open the HDF5 file and only load a fraction of question groups.
        with h5py.File(annotations_h5_path, 'r') as hf:
            q_group = hf["questions"]
            all_keys = sorted(list(q_group.keys()))
            num_to_use = int(len(all_keys) * data_fraction)
            selected_keys = all_keys[:num_to_use]
            print(f"Using {data_fraction*100:.1f}% of question groups: {len(selected_keys)} out of {len(all_keys)}")
            for q_key in selected_keys:
                grp = q_group[q_key]
                question = {}
                for key in grp.keys():
                    val = grp[key][()]
                    if isinstance(val, bytes):
                        try:
                            val = val.decode('utf-8')
                        except Exception:
                            val = str(val)
                    try:
                        question[key] = json.loads(val)
                    except Exception:
                        question[key] = val
                # For each annotated_program step, create a sample.
                image_index = question["image_index"]
                for step in question.get("annotated_program", []):
                    out_str = step.get("output_values", "")
                    tokens = tokenize_output(out_str)
                    if len(tokens) > 0 and (len(tokens) % 6 == 0):
                        output_type = 0  # spatial
                        groups = [tokens[i:i+6] for i in range(0, len(tokens), 6)]
                        # Remove the separator tokens (first and last tokens)
                        box_tokens = [group[1:5] for group in groups]
                        output_tensor = torch.tensor(box_tokens, dtype=torch.float32)
                    else:
                        output_type = 1  # nonspatial
                        output_tensor = torch.tensor(tokens, dtype=torch.float32)
                    sample = {
                        "image_index": image_index,
                        "output_values": output_tensor,
                        "output_type": output_type,
                    }
                    self.samples.append(sample)
        print(f"Built {len(self.samples)} training samples from annotated questions.")
        # Count and print how many samples have zero elements in their output.
        zero_count = sum(1 for sample in self.samples if sample["output_values"].numel() == 0)
        print(f"Number of samples with zero output: {zero_count} out of {len(self.samples)}")

        # Open image features file.
        self.img_h5 = h5py.File(image_features_h5_path, 'r')
        self.img_features = self.img_h5["features"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_idx = sample["image_index"]
        img_feat = self.img_features[img_idx]  # shape: [1024, 14, 14]
        img_feat = torch.tensor(img_feat, dtype=torch.float32)
        return {
            "img_features": img_feat,
            "output_values": sample["output_values"],
            "output_type": sample["output_type"]
        }

#############################################
# Model: Hierarchical Multi-Head Generator (Tokenized Version)
#############################################
class HierarchicalMultiHeadGenerator(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, max_inner_steps=10):
        """
        The model predicts tokenized outputs.
        It contains:
          - A type prediction head (outputs logits for two classes):
              0: spatial (bounding box sequence)
              1: nonspatial (a single token)
          - A spatial branch that autoregressively generates groups of 4 tokens (the numeric part of a bounding box).
          - A nonspatial branch that predicts a single token.
        """
        super().__init__()
        self.d_model = d_model
        self.max_inner_steps = max_inner_steps

        self.image_proj = nn.Linear(1024, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.start_token = nn.Parameter(torch.randn(d_model))
        self.type_head = nn.Linear(d_model, 2)

        self.bbox_embedding = nn.Linear(4, d_model)
        self.bbox_out = nn.Linear(d_model, 4)
        self.stop_out = nn.Linear(d_model, 1)

        self.nonspatial_out = nn.Linear(d_model, 1)

    def forward(self, img_features, gt_outputs=None, teacher_forcing=True, gt_type=None):
        # Encode image features.
        img_feat_2d = img_features.view(1024, -1).transpose(0, 1)  # [196, 1024]
        enc_input = self.image_proj(img_feat_2d)  # [196, d_model]
        enc_input = enc_input.unsqueeze(1)  # [196, 1, d_model]
        memory = self.encoder(enc_input)

        # Global representation from start token.
        dec_input_start = self.start_token.unsqueeze(0).unsqueeze(1)  # [1, 1, d_model]
        dec_output_start = self.decoder(dec_input_start, memory)
        global_rep = dec_output_start.squeeze(0).squeeze(0)  # [d_model]
        type_logits = self.type_head(global_rep)  # [2]
        type_probs = torch.softmax(type_logits, dim=-1)
        if gt_type is None:
            pred_type = torch.argmax(type_probs).item()
        else:
            pred_type = gt_type

        if pred_type == 0:
            # For teacher forcing: if gt_outputs has an extra batch dimension, remove it.
            if teacher_forcing and gt_outputs.dim() == 3:
                gt_outputs = gt_outputs[0]
            if teacher_forcing:
                T = gt_outputs.size(0)
                start = self.start_token.unsqueeze(0)  # [1, d_model]
                emb_boxes = self.bbox_embedding(gt_outputs)  # [T, d_model]
                dec_input = torch.cat([start, emb_boxes], dim=0)  # [T+1, d_model]
            else:
                dec_input = self.start_token.unsqueeze(0)
                T = self.max_inner_steps
            dec_input = dec_input.unsqueeze(1)  # [L, 1, d_model]
            L = dec_input.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(dec_input.device)
            dec_output = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
            dec_output = dec_output[1:].squeeze(1)  # [L-1, d_model]
            pred_bboxes = self.bbox_out(dec_output)
            pred_stop_logits = self.stop_out(dec_output)
            branch_outputs = (pred_bboxes, pred_stop_logits)
        else:
            dec_input = self.start_token.unsqueeze(0).unsqueeze(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(0)).to(dec_input.device)
            dec_output = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
            dec_output = dec_output.squeeze(0).squeeze(0)
            pred_value = self.nonspatial_out(dec_output)
            branch_outputs = pred_value

        return type_logits, branch_outputs

    def generate(self, img_features, stop_threshold=0.5):
        img_feat_2d = img_features.view(1024, -1).transpose(0, 1)
        enc_input = self.image_proj(img_feat_2d)
        enc_input = enc_input.unsqueeze(1)
        memory = self.encoder(enc_input)
        dec_input_start = self.start_token.unsqueeze(0).unsqueeze(1)
        dec_output_start = self.decoder(dec_input_start, memory)
        global_rep = dec_output_start.squeeze(0).squeeze(0)
        type_logits = self.type_head(global_rep)
        type_probs = torch.softmax(type_logits, dim=-1)
        pred_type = torch.argmax(type_probs).item()

        if pred_type == 0:
            dec_input = self.start_token.unsqueeze(0).unsqueeze(1)
            generated_boxes = []
            for t in range(self.max_inner_steps):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(0)).to(dec_input.device)
                dec_output = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
                last_output = dec_output[-1, 0, :]
                bbox_pred = self.bbox_out(last_output)
                stop_logit = self.stop_out(last_output)
                stop_prob = torch.sigmoid(stop_logit).item()
                # Wrap the predicted 4 tokens with separator tokens (19: "[", 24: "]").
                generated_box = [19] + bbox_pred.detach().cpu().tolist() + [24]
                generated_boxes.append(generated_box)
                if stop_prob >= stop_threshold:
                    break
                next_input = self.bbox_embedding(bbox_pred.unsqueeze(0))
                next_input = next_input.unsqueeze(1)
                dec_input = torch.cat([dec_input, next_input], dim=0)
            return generated_boxes
        else:
            dec_input = self.start_token.unsqueeze(0).unsqueeze(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(0)).to(dec_input.device)
            dec_output = self.decoder(dec_input, memory, tgt_mask=tgt_mask)
            dec_output = dec_output.squeeze(0).squeeze(0)
            pred_value = self.nonspatial_out(dec_output)
            return pred_value.detach().cpu().tolist()

#############################################
# Training Loop
#############################################
def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths to the separated questions HDF5 file and image features file.
    annotations_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/annotated_questions_separated.h5"  # Update if needed.
    image_features_path = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"

    # Use only a fraction of the data.
    data_fraction = 0.01
    dataset = ClevrStepsDatasetNew(annotations_path, image_features_path, data_fraction=data_fraction)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Compute maximum number of bounding boxes among spatial samples.
    max_bbox_count = 0
    for sample in dataset.samples:
        if sample["output_type"] == 0:
            bbox_count = sample["output_values"].size(0)
            if bbox_count > max_bbox_count:
                max_bbox_count = bbox_count
    print(f"Maximum bounding boxes in dataset: {max_bbox_count}")

    # Initialize model with max_inner_steps set to max_bbox_count.
    model = HierarchicalMultiHeadGenerator(d_model=256, nhead=4, num_layers=2, max_inner_steps=max_bbox_count)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()            # For regression (both branches)
    bce_loss = nn.BCEWithLogitsLoss()    # For stop signal (spatial branch)
    ce_loss = nn.CrossEntropyLoss()      # For type prediction

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = 2
    for epoch in range(num_epochs):
        for step_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            img_features = batch["img_features"].to(device)
            gt_outputs = batch["output_values"].to(device)
            gt_type = batch["output_type"][0]  # 0: spatial, 1: nonspatial

            # Skip samples with empty ground truth.
            if gt_outputs.numel() == 0:
                print(f"Step {step_idx}: Skipping sample due to empty ground truth.")
                continue

            type_logits, branch_outputs = model(img_features[0],
                                                gt_outputs=gt_outputs,
                                                teacher_forcing=True,
                                                gt_type=gt_type)
            gt_type_tensor = torch.tensor([gt_type], dtype=torch.long, device=device)
            type_loss = ce_loss(type_logits.unsqueeze(0), gt_type_tensor)

            predicted_type = torch.argmax(type_logits).item()
            if predicted_type != gt_type:
                print(f"Step {step_idx}: Type mismatch, predicted: {predicted_type}, gt: {gt_type}")

            if gt_type == 0:
                pred_bboxes, pred_stop_logits = branch_outputs
                bbox_loss = mse_loss(pred_bboxes, gt_outputs)
                stop_target = torch.zeros(pred_stop_logits.size(), device=device)
                if pred_stop_logits.size(0) > 0:
                    stop_target[-1] = 1.0
                stop_loss = bce_loss(pred_stop_logits, stop_target)
                branch_loss = bbox_loss + stop_loss
                if (bbox_loss.item() + stop_loss.item()) > 1000:
                    print(f"Step {step_idx}: High spatial loss - bbox_loss: {bbox_loss.item():.4f}, stop_loss: {stop_loss.item():.4f}")
            else:
                branch_loss = mse_loss(branch_outputs.squeeze(), gt_outputs.squeeze())
                if branch_loss.item() > 1000:
                    print(f"Step {step_idx}: High nonspatial loss: {branch_loss.item():.4f}")

            loss = type_loss + branch_loss
            loss.backward()
            optimizer.step()

            if step_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {step_idx}, Total Loss={loss.item():.4f}, Type Loss={type_loss.item():.4f}, Branch Loss={branch_loss.item():.4f}")

            if step_idx % 1000 == 0 and step_idx > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}_step{step_idx}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        epoch_checkpoint = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), epoch_checkpoint)
        print(f"Epoch {epoch} checkpoint saved at {epoch_checkpoint}")

    print("Training completed!")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    model.eval()
    sample = next(iter(dataloader))
    img_features_sample = sample["img_features"][0].to(device)
    generated_output = model.generate(img_features_sample, stop_threshold=0.5)
    print("Generated output:", generated_output)

if __name__ == "__main__":
    train_model()
