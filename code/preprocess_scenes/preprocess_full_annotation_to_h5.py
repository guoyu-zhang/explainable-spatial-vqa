import os
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

#############################################
# Dataset: ClevrStepsDataset
#############################################
class ClevrStepsDataset(Dataset):
    def __init__(self, annotations_h5_path, image_features_h5_path):
        super().__init__()
        self.annotations_h5_path = annotations_h5_path
        self.image_features_h5_path = image_features_h5_path

        # Open the annotations HDF5 file and get the "questions" group.
        self.annotations_h5 = h5py.File(self.annotations_h5_path, "r")
        self.questions_group = self.annotations_h5["questions"]

        # Open the image features HDF5 file.
        self.img_feat_h5 = h5py.File(self.image_features_h5_path, "r")
        self.img_features_dataset = self.img_feat_h5["features"]  # shape: [70000, 1024, 14, 14]

        self.training_steps = []
        self._build_training_steps()

    def _build_training_steps(self):
        """
        Iterate over each question and each program step; each step becomes one training sample.
        Each sample includes an output bounding-box sequence (a list of boxes, each a list of 4 numbers).
        """
        for q_key in self.questions_group.keys():
            q_group = self.questions_group[q_key]
            image_index = int(q_group["image_index"][()])
            question_index = int(q_group["question_index"][()])
            question_str = q_group["question"][()].decode("utf-8")
            answer_str = q_group["answer"][()].decode("utf-8")
            annotated_program = q_group["annotated_program"]
            # Get step keys (e.g., "step_00", "step_01", ...)
            step_keys = sorted(annotated_program.keys())
            for sk in step_keys:
                step_grp = annotated_program[sk]
                # (Chain-of-thought tokens are loaded but not used for box generation in this model.)
                chain_str = step_grp["chain_of_thought"][()].decode("utf-8")
                chain_list = json.loads(chain_str)
                in_val_str = step_grp["input_values"][()].decode("utf-8")
                input_vals = json.loads(in_val_str)
                out_val_str = step_grp["output_values"][()].decode("utf-8")
                output_vals = json.loads(out_val_str)  # Expected to be a list of boxes, e.g. [[0.5,0.2,0.6,0.4], ...]
                sample = {
                    "image_index": image_index,
                    "question_index": question_index,
                    "question_str": question_str,
                    "answer_str": answer_str,
                    "chain_of_thought_tokens": chain_list,
                    "input_values": input_vals,
                    "output_values": output_vals,  # Ground truth bounding boxes (variable length)
                }
                self.training_steps.append(sample)
        print(f"Built {len(self.training_steps)} training steps.")

    def __len__(self):
        return len(self.training_steps)

    def __getitem__(self, idx):
        step_info = self.training_steps[idx]
        # Load image features based on image_index (shape: [1024, 14, 14])
        img_idx = step_info["image_index"]
        img_features = self.img_features_dataset[img_idx]
        img_features = torch.tensor(img_features, dtype=torch.float32)

        # (Optionally, load chain tokens if needed for other tasks)
        chain_tokens = step_info["chain_of_thought_tokens"]
        if chain_tokens and isinstance(chain_tokens[0], int):
            chain_tokens = torch.tensor(chain_tokens, dtype=torch.long)
        else:
            chain_tokens = torch.tensor([], dtype=torch.long)

        # Process output_values: ground truth bbox sequence (shape: [T, 4])
        output_values = step_info["output_values"]
        if isinstance(output_values, list) and len(output_values) > 0 and isinstance(output_values[0], list):
            output_values = torch.tensor(output_values, dtype=torch.float32)
        else:
            output_values = torch.tensor([], dtype=torch.float32)

        # Return the sample.
        return {
            "img_features": img_features,           # Tensor [1024, 14, 14]
            "chain_tokens": chain_tokens,           # Tensor [seq_len] (if needed)
            "output_values": output_values,         # Tensor [T, 4], T is variable
            "question_str": step_info["question_str"],
            "answer_str": step_info["answer_str"],
            "question_index": step_info["question_index"],
        }

#############################################
# Model: Hierarchical Autoregressive BBox Generator
#############################################
class HierarchicalBBoxGenerator(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, max_inner_steps=10):
        """
        This model generates a variable-length sequence of bounding boxes for a given program step.
        It first encodes image features, then uses an autoregressive inner decoder to generate boxes.
        At each inner decoding step, it outputs:
          - A predicted bounding box (4 continuous values)
          - A stop signal (scalar; 0 means "continue", 1 means "stop")
        """
        super().__init__()
        self.d_model = d_model
        self.max_inner_steps = max_inner_steps

        # Encoder for image features.
        self.image_proj = nn.Linear(1024, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Inner autoregressive decoder (Transformer-based).
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Learned start token for the inner decoder.
        self.start_token = nn.Parameter(torch.randn(d_model))

        # BBox embedding: projects a continuous bounding box (4-dim) into d_model space.
        self.bbox_embedding = nn.Linear(4, d_model)

        # Output heads:
        # Predict a bounding box from a decoder output.
        self.bbox_out = nn.Linear(d_model, 4)
        # Predict a stop signal (logit); apply sigmoid externally.
        self.stop_out = nn.Linear(d_model, 1)

    def forward(self, img_features, gt_bboxes, teacher_forcing=True):
        """
        Args:
            img_features: Tensor of shape [1024, 14, 14] for one sample.
            gt_bboxes: Tensor of shape [T, 4] (ground truth sequence of boxes) if teacher_forcing=True.
            teacher_forcing: If True, use the ground truth sequence for the inner decoder.
        Returns:
            pred_bboxes: Tensor of shape [T, 4] – predicted bounding boxes for each inner step.
            pred_stop_logits: Tensor of shape [T, 1] – predicted stop logits for each inner step.
        """
        # --- Encode Image Features ---
        # Flatten image features: [1024, 14, 14] -> [196, 1024]
        img_feat_2d = img_features.view(1024, -1).transpose(0, 1)  # [196, 1024]
        enc_input = self.image_proj(img_feat_2d)  # [196, d_model]
        enc_input = enc_input.unsqueeze(1)  # [196, 1, d_model]
        memory = self.encoder(enc_input)      # [196, 1, d_model]

        if teacher_forcing:
            T = gt_bboxes.size(0)  # ground truth sequence length.
            # Prepare inner decoder input:
            # dec_input[0] = start token; for t>=1, use bbox_embedding(gt_bboxes[t-1])
            start = self.start_token.unsqueeze(0)  # [1, d_model]
            emb_boxes = self.bbox_embedding(gt_bboxes)  # [T, d_model]
            dec_input = torch.cat([start, emb_boxes], dim=0)  # [T+1, d_model]
        else:
            # In inference, start with start token only.
            dec_input = self.start_token.unsqueeze(0)  # [1, d_model]
            T = self.max_inner_steps

        # Add batch dimension: [L, 1, d_model]
        dec_input = dec_input.unsqueeze(1)
        L = dec_input.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(dec_input.device)
        dec_output = self.decoder(dec_input, memory, tgt_mask=tgt_mask)  # [L, 1, d_model]

        # We use decoder outputs from time steps 1..L as predictions.
        # That yields a sequence of length L-1. (During teacher forcing, L = T+1)
        dec_output = dec_output[1:]  # [L-1, 1, d_model]
        dec_output = dec_output.squeeze(1)  # [L-1, d_model]

        pred_bboxes = self.bbox_out(dec_output)       # [L-1, 4]
        pred_stop_logits = self.stop_out(dec_output)    # [L-1, 1]

        return pred_bboxes, pred_stop_logits

    def generate(self, img_features, max_steps=10, stop_threshold=0.5):
        """
        Autoregressively generate bounding boxes until the stop signal exceeds the threshold or max_steps is reached.
        Args:
            img_features: Tensor of shape [1024, 14, 14].
            max_steps: Maximum number of inner decoding steps.
            stop_threshold: Threshold for stop signal (after sigmoid) to terminate generation.
        Returns:
            generated_boxes: List of predicted bounding boxes (each a list of 4 floats).
        """
        # Encode image features.
        img_feat_2d = img_features.view(1024, -1).transpose(0, 1)  # [196, 1024]
        enc_input = self.image_proj(img_feat_2d)  # [196, d_model]
        enc_input = enc_input.unsqueeze(1)  # [196, 1, d_model]
        memory = self.encoder(enc_input)      # [196, 1, d_model]

        # Initialize decoder input with start token.
        dec_input = self.start_token.unsqueeze(0).unsqueeze(1)  # [1, 1, d_model]
        generated_boxes = []

        for t in range(max_steps):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(0)).to(dec_input.device)
            dec_output = self.decoder(dec_input, memory, tgt_mask=tgt_mask)  # [L, 1, d_model]
            last_output = dec_output[-1, 0, :]  # [d_model]
            bbox_pred = self.bbox_out(last_output)  # [4]
            stop_logit = self.stop_out(last_output)  # [1]
            stop_prob = torch.sigmoid(stop_logit).item()
            generated_boxes.append(bbox_pred.detach().cpu().tolist())

            if stop_prob >= stop_threshold:
                break

            # Prepare next decoder input using the embedding of the predicted bbox.
            next_input = self.bbox_embedding(bbox_pred.unsqueeze(0))  # [1, d_model]
            next_input = next_input.unsqueeze(1)  # [1, 1, d_model]
            dec_input = torch.cat([dec_input, next_input], dim=0)

        return generated_boxes

#############################################
# Training Loop with MPS Device Support
#############################################
def train_model():
    # Set device: use MPS if available, else fallback to CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths to HDF5 files.
    annotations_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/output_full_annotations.h5"
    image_features_path = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"

    # Create dataset and dataloader (batch_size=1 for simplicity).
    dataset = ClevrStepsDataset(annotations_path, image_features_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Instantiate model and move to device.
    model = HierarchicalBBoxGenerator(d_model=256, nhead=4, num_layers=2, max_inner_steps=10)
    model.to(device)
    model.train()

    # Define optimizer and loss functions.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()         # For bounding box regression.
    bce_loss = nn.BCEWithLogitsLoss()  # For stop signal prediction.

    num_epochs = 2  # For demonstration.
    for epoch in range(num_epochs):
        for step_idx, batch in enumerate(dataloader):
            # Move tensors to device.
            img_features = batch["img_features"].to(device)   # [1, 1024, 14, 14]
            gt_boxes = batch["output_values"].to(device)        # [1, T, 4]
            # Skip samples with no valid ground truth.
            if gt_boxes.dim() != 3 or gt_boxes.size(1) != 4 or gt_boxes.size(0) == 0:
                continue
            gt_boxes = gt_boxes[0]  # [T, 4] for batch_size=1

            optimizer.zero_grad()
            # Forward pass with teacher forcing.
            pred_boxes, pred_stop_logits = model(img_features[0], gt_bboxes=gt_boxes, teacher_forcing=True)
            # pred_boxes: [T_pred, 4], pred_stop_logits: [T_pred, 1]
            # In teacher forcing, T_pred = T (if using ground truth length).

            T = gt_boxes.size(0)
            # For bounding box loss: compare predicted boxes with ground truth.
            bbox_loss = mse_loss(pred_boxes, gt_boxes)

            # For stop signal loss: target = 0 for time steps 0...T-2, and 1 for time step T-1.
            stop_target = torch.zeros(pred_stop_logits.size(), device=device)
            if pred_stop_logits.size(0) > 0:
                stop_target[-1] = 1.0
            stop_loss = bce_loss(pred_stop_logits, stop_target)

            loss = bbox_loss + stop_loss
            loss.backward()
            optimizer.step()

            if step_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {step_idx}, Loss={loss.item():.4f}")

    print("Training completed!")

    # Optionally, run generation on one sample.
    model.eval()
    sample = next(iter(dataloader))
    img_features_sample = sample["img_features"][0].to(device)
    generated_boxes = model.generate(img_features_sample, max_steps=10, stop_threshold=0.5)
    print("Generated bounding boxes:", generated_boxes)

#############################################
# Main
#############################################
if __name__ == "__main__":
    train_model()
