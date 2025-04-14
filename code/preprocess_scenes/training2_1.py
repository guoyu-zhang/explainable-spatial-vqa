import os
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

##############################
# Helper Functions
##############################

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.
    Each box is defined as [x, y, w, h] with (x,y) as center.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert center coordinates to corner coordinates.
    box1_x1 = x1 - w1 / 2
    box1_y1 = y1 - h1 / 2
    box1_x2 = x1 + w1 / 2
    box1_y2 = y1 + h1 / 2

    box2_x1 = x2 - w2 / 2
    box2_y1 = y2 - h2 / 2
    box2_x2 = x2 + w2 / 2
    box2_y2 = y2 + h2 / 2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def parse_bbox_tokens(token_str, start_token="19", end_token="24", num_coords=4):
    """
    Parse a string of tokens (separated by spaces) into a list of bounding boxes.
    Each box is assumed to start with start_token and end with end_token,
    with num_coords coordinate tokens in between.
    For example, "19 22 23 28 22 24" produces one box: [22, 23, 28, 22].
    Returns a list of boxes (each a list of floats).
    """
    tokens = token_str.strip().split()
    boxes = []
    i = 0
    while i < len(tokens):
        if tokens[i] == start_token:
            if i + num_coords + 1 < len(tokens) and tokens[i + num_coords + 1] == end_token:
                coords = [float(tok) for tok in tokens[i+1:i+1+num_coords]]
                boxes.append(coords)
                i += num_coords + 2
            else:
                break
        else:
            i += 1
    return boxes

##############################
# Variable-Length Bounding Box Decoder
##############################

class BBoxDecoder(nn.Module):
    def __init__(self, hidden_dim, box_dim=4, max_steps=10):
        """
        Autoregressive bounding box decoder.
        Uses an LSTMCell to predict one box at a time along with a stop flag.
        - hidden_dim: dimension of the decoder input (and hidden state).
        - box_dim: number of coordinates (typically 4).
        - max_steps: maximum number of decoding steps to avoid infinite loops.
        """
        super(BBoxDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.box_dim = box_dim
        self.max_steps = max_steps
        self.lstm_cell = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        # A learned start token embedding for initiating the decoder.
        self.start_token = nn.Parameter(torch.randn(hidden_dim))
        # Heads for regression and stop prediction.
        self.box_out = nn.Linear(hidden_dim, box_dim)
        self.stop_out = nn.Linear(hidden_dim, 2)  # two classes: continue (0) or stop (1)
        # Projection layer to map a box (or teacher box) from 4-dim to hidden_dim.
        self.input_proj = nn.Linear(box_dim, hidden_dim)

    def forward(self, shared_repr, teacher_boxes=None, teacher_forcing_ratio=0.5):
        """
        shared_repr: Tensor (B, hidden_dim) from the encoder.
        teacher_boxes: List (length B) where each element is a tensor of shape (T_i, 4)
                       containing ground truth boxes (if available).
        teacher_forcing_ratio: probability of using teacher forcing.
        Returns:
            A list (length B) of lists; each inner list is of length <= max_steps containing
            tuples (box_pred, stop_logits) for each decoding step.
        """
        B = shared_repr.size(0)
        outputs = [[] for _ in range(B)]
        # Initialize hidden and cell states.
        hx = shared_repr  # (B, hidden_dim)
        cx = torch.zeros_like(hx)
        # Initial input: learned start token.
        input_t = self.start_token.unsqueeze(0).expand(B, -1)  # (B, hidden_dim)

        for t in range(self.max_steps):
            hx, cx = self.lstm_cell(input_t, (hx, cx))
            # Predict box coordinates.
            box_pred = self.box_out(hx)  # (B, box_dim)
            # Predict stop flag logits.
            stop_logits = self.stop_out(hx)  # (B, 2)

            # Save outputs.
            for i in range(B):
                outputs[i].append((box_pred[i], stop_logits[i]))

            # Decide next input for each sample.
            next_inputs = []
            for i in range(B):
                teacher_box = teacher_boxes[i] if teacher_boxes is not None else None
                # Use teacher forcing if teacher box exists and within sequence length.
                if teacher_box is not None and t < teacher_box.size(0) and np.random.rand() < teacher_forcing_ratio:
                    next_in = self.input_proj(teacher_box[t].unsqueeze(0)).squeeze(0)
                else:
                    next_in = self.input_proj(box_pred[i].unsqueeze(0)).squeeze(0)
                next_inputs.append(next_in)
            input_t = torch.stack(next_inputs, dim=0)
        return outputs

##############################
# Dataset Definition
##############################

class CLEVRDataset(Dataset):
    def __init__(self, questions_file, features_file, vocab_file, max_seq_len=20, subset_frac=1.0):
        """
        questions_file: path to flattened_questions.h5
        features_file: path to train_features.h5
        vocab_file: path to vocab.json
        max_seq_len: maximum length for input token sequences.
        subset_frac: fraction of the data to load (e.g., 0.1 for 10% of data)
        """
        self.questions_file = questions_file
        self.features_file_path = features_file
        self.max_seq_len = max_seq_len

        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        # Create a reverse mapping: number -> string.
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

        # Build specialized function token sets using the numerical tokens.
        self.bbox_function_tokens = set()
        for token, string in self.rev_vocab.items():
            if string.startswith("scene") or string.startswith("relate[") or string.startswith("filter_") or string.startswith("same_") or string == "unique":
                self.bbox_function_tokens.add(token)

        self.integer_function_tokens = {self.vocab["count"]} if "count" in self.vocab else set()

        self.boolean_function_tokens = set()
        for key in ["exist", "equal_integer", "less_than", "greater_than"]:
            if key in self.vocab:
                self.boolean_function_tokens.add(self.vocab[key])

        self.size_function_tokens = {self.vocab["query_size"]} if "query_size" in self.vocab else set()
        self.color_function_tokens = {self.vocab["query_color"]} if "query_color" in self.vocab else set()
        self.shape_function_tokens = {self.vocab["query_shape"]} if "query_shape" in self.vocab else set()
        self.material_function_tokens = {self.vocab["query_material"]} if "query_material" in self.vocab else set()

        self.samples = []
        with h5py.File(self.questions_file, "r") as f:
            # List all keys.
            all_keys = list(f.keys())
            # Select only a subset of keys based on subset_frac.
            if subset_frac < 1.0:
                num_keys = int(len(all_keys) * subset_frac)
                keys = all_keys[:num_keys]
            else:
                keys = all_keys

            for key in keys:
                group = f[key]
                prog = json.loads(group["annotated_program"][()])
                image_index = int(group["image_index"][()])
                for step in prog:
                    # Get the function token from the step (which is now a number as a string)
                    function_str = step["function"]
                    function_token = int(function_str)
                    input_values = step["input_values"].strip()
                    output_values = step["output_values"].strip()
                    if output_values == "":
                        continue
                    tokens_out = output_values.split()
                    # Determine target type using the numerical tokens.
                    if function_token in self.bbox_function_tokens:
                        target_type = "bbox"
                    elif function_token in self.integer_function_tokens:
                        target_type = "integer"
                    elif function_token in self.boolean_function_tokens:
                        target_type = "boolean"
                    elif function_token in self.size_function_tokens:
                        target_type = "size"
                    elif function_token in self.color_function_tokens:
                        target_type = "color"
                    elif function_token in self.shape_function_tokens:
                        target_type = "shape"
                    elif function_token in self.material_function_tokens:
                        target_type = "material"
                    else:
                        target_type = "vocab"  # default uses full vocabulary

                    if input_values != "":
                        input_token_ids = [int(tok) for tok in input_values.split()]
                    else:
                        input_token_ids = []

                    sample = {
                        "function_str": function_str,
                        "function_token": function_token,
                        "input_tokens": torch.tensor(input_token_ids, dtype=torch.long),
                        "output_values": output_values,
                        "target_type": target_type,
                        "image_index": image_index
                    }
                    if target_type == "bbox":
                        boxes = parse_bbox_tokens(output_values, start_token="19", end_token="24")
                        sample["target_boxes_seq"] = torch.tensor(boxes, dtype=torch.float32)
                    else:
                        sample["target_label"] = int(tokens_out[0])
                    self.samples.append(sample)
        self.features_file = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.features_file is None:
            self.features_file = h5py.File(self.features_file_path, "r")
        image_index = sample["image_index"]
        image_feat = torch.tensor(self.features_file["features"][image_index], dtype=torch.float32)
        sample["image_feat"] = image_feat
        return sample

def clevr_collate_fn(batch):
    """
    Collate function.
    Pads input_tokens; bbox targets are kept as variable-length lists.
    """
    function_tokens = torch.tensor([s["function_token"] for s in batch], dtype=torch.long)
    image_feats = torch.stack([s["image_feat"] for s in batch])
    seqs = [s["input_tokens"] for s in batch]
    padded_inputs = pad_sequence(seqs, batch_first=True, padding_value=0)
    input_lengths = torch.tensor([len(s["input_tokens"]) for s in batch], dtype=torch.long)

    target_types = [s["target_type"] for s in batch]
    target_labels = []
    bbox_targets = []  # For bbox, store variable-length tensor per sample.
    for s in batch:
        if s["target_type"] == "bbox":
            bbox_targets.append(s["target_boxes_seq"])
            target_labels.append(-1)  # dummy value for bbox
        else:
            bbox_targets.append(None)
            target_labels.append(s["target_label"])
    target_labels = torch.tensor(target_labels, dtype=torch.long)

    return {
        "function_tokens": function_tokens,
        "input_tokens": padded_inputs,
        "input_lengths": input_lengths,
        "image_feats": image_feats,
        "target_types": target_types,
        "bbox_targets": bbox_targets,
        "target_labels": target_labels
    }

##############################
# Multi-Head Model Definition
##############################

class MultiHeadModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, image_feat_dim=1024, max_bbox_steps=10):
        """
        Multi-head model with a shared encoder and separate output heads.
        """
        super(MultiHeadModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.image_fc = nn.Linear(image_feat_dim * 14 * 14, hidden_dim)
        self.fc_shared = nn.Linear(hidden_dim * 2, hidden_dim)
        self.max_bbox_steps = max_bbox_steps

        # Variable-length bounding box decoder.
        self.bbox_decoder = BBoxDecoder(hidden_dim, box_dim=4, max_steps=max_bbox_steps)
        # Other heads:
        self.integer_head = nn.Linear(hidden_dim, 11)   # For count, 0-10.
        self.boolean_head = nn.Linear(hidden_dim, 2)
        self.size_head = nn.Linear(hidden_dim, 2)
        self.color_head = nn.Linear(hidden_dim, 8)
        self.shape_head = nn.Linear(hidden_dim, 3)
        self.material_head = nn.Linear(hidden_dim, 2)
        # New head for full-vocab outputs.
        self.vocab_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, function_tokens, input_tokens, image_feats, bbox_teacher_boxes=None):
        """
        Forward pass.
        bbox_teacher_boxes: List (length B) of teacher boxes for bbox decoding (if available).
        """
        # Text encoding.
        func_embed = self.embedding(function_tokens).unsqueeze(1)
        input_embed = self.embedding(input_tokens)
        text_input = torch.cat([func_embed, input_embed], dim=1)
        _, (hidden, _) = self.text_encoder(text_input)
        text_repr = hidden[-1]

        # Image encoding.
        B = image_feats.size(0)
        image_feats_flat = image_feats.view(B, -1)
        image_repr = F.relu(self.image_fc(image_feats_flat))
        combined = torch.cat([text_repr, image_repr], dim=1)
        shared_repr = F.relu(self.fc_shared(combined))  # (B, hidden_dim)

        # Decode bounding box outputs.
        bbox_out = self.bbox_decoder(shared_repr, teacher_boxes=bbox_teacher_boxes)
        integer_out = self.integer_head(shared_repr)
        boolean_out = self.boolean_head(shared_repr)
        size_out = self.size_head(shared_repr)
        color_out = self.color_head(shared_repr)
        shape_out = self.shape_head(shared_repr)
        material_out = self.material_head(shared_repr)
        vocab_out = self.vocab_head(shared_repr)

        return {
            "bbox": bbox_out,  # List (B elements) of lists of (box_pred, stop_logits) tuples.
            "integer": integer_out,
            "boolean": boolean_out,
            "size": size_out,
            "color": color_out,
            "shape": shape_out,
            "material": material_out,
            "vocab": vocab_out
        }

##############################
# Training Loop
##############################

def train_model(model, dataloader, optimizer, num_epochs, device):
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    smooth_l1_loss = nn.SmoothL1Loss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            function_tokens = batch["function_tokens"].to(device)
            input_tokens = batch["input_tokens"].to(device)
            image_feats = batch["image_feats"].to(device)
            target_types = batch["target_types"]
            target_labels = batch["target_labels"].to(device)
            bbox_targets = batch["bbox_targets"]  # List of variable-length tensors (or None)

            optimizer.zero_grad()

            # Prepare teacher boxes for bbox decoding.
            bbox_teacher_boxes = []
            for i, t in enumerate(target_types):
                if t == "bbox" and bbox_targets[i] is not None:
                    teacher = bbox_targets[i].to(device)  # (T, 4)
                    bbox_teacher_boxes.append(teacher)
                else:
                    bbox_teacher_boxes.append(None)

            outputs = model(function_tokens, input_tokens, image_feats, bbox_teacher_boxes=bbox_teacher_boxes)

            loss = 0.0
            B = function_tokens.size(0)
            for i in range(B):
                ttype = target_types[i]
                if ttype == "bbox":
                    pred_seq = outputs["bbox"][i]  # List of (box_pred, stop_logits)
                    gt_boxes = bbox_targets[i].to(device)  # (T, 4)
                    T = gt_boxes.size(0)
                    gt_stop = [0] * T + [1]  # 0 for continue, 1 for stop
                    seq_steps = min(len(pred_seq), T + 1)
                    loss_bbox = 0.0
                    loss_stop = 0.0
                    for t in range(seq_steps):
                        box_pred, stop_logits = pred_seq[t]
                        if t < T:
                            loss_bbox += smooth_l1_loss(box_pred, gt_boxes[t])
                        loss_stop += ce_loss(stop_logits.unsqueeze(0),
                                              torch.tensor([gt_stop[t]], dtype=torch.long, device=device))
                    loss_sample = (loss_bbox + loss_stop) / seq_steps
                    loss += loss_sample
                elif ttype == "integer":
                    pred = outputs["integer"][i].unsqueeze(0)
                    loss += ce_loss(pred, target_labels[i].unsqueeze(0))
                elif ttype == "boolean":
                    pred = outputs["boolean"][i].unsqueeze(0)
                    loss += ce_loss(pred, target_labels[i].unsqueeze(0))
                elif ttype == "size":
                    pred = outputs["size"][i].unsqueeze(0)
                    loss += ce_loss(pred, target_labels[i].unsqueeze(0))
                elif ttype == "color":
                    pred = outputs["color"][i].unsqueeze(0)
                    loss += ce_loss(pred, target_labels[i].unsqueeze(0))
                elif ttype == "shape":
                    pred = outputs["shape"][i].unsqueeze(0)
                    loss += ce_loss(pred, target_labels[i].unsqueeze(0))
                elif ttype == "material":
                    pred = outputs["material"][i].unsqueeze(0)
                    loss += ce_loss(pred, target_labels[i].unsqueeze(0))
                elif ttype == "vocab":
                    pred = outputs["vocab"][i].unsqueeze(0)
                    loss += ce_loss(pred, target_labels[i].unsqueeze(0))
                else:
                    pred = outputs["integer"][i].unsqueeze(0)
                    loss += ce_loss(pred, target_labels[i].unsqueeze(0))
            loss = loss / B
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

##############################
# Inference Function
##############################
def run_inference(model, sample, device, vocab):
    """
    Run inference on a single sample and print the outputs.
    The sample is a dict from the dataset; we assume teacher forcing is off (i.e. bbox_teacher_boxes=None).
    """
    model.eval()
    with torch.no_grad():
        # Convert the function token from int to tensor.
        function_tokens = torch.tensor([sample["function_token"]], dtype=torch.long).to(device)
        # The input tokens should already be a tensor.
        input_tokens = sample["input_tokens"].unsqueeze(0).to(device)
        image_feats = sample["image_feat"].unsqueeze(0).to(device)
        # For inference, we set teacher boxes to None.
        outputs = model(function_tokens, input_tokens, image_feats, bbox_teacher_boxes=[None])
    
    print("Function:", sample["function_str"])
    print("Target type:", sample["target_type"])
    
    if sample["target_type"] == "bbox":
        # For bounding boxes, print each decoding step.
        bbox_preds = outputs["bbox"][0]  # List of (box_pred, stop_logits)
        for i, (box_pred, stop_logits) in enumerate(bbox_preds):
            stop_class = torch.argmax(stop_logits).item()
            print(f"Step {i}: Box prediction: {box_pred.cpu().numpy()}, Stop flag: {stop_class}")
    elif sample["target_type"] == "integer":
        integer_pred = outputs["integer"]
        predicted_int = torch.argmax(integer_pred, dim=1).item()
        print("Predicted integer:", predicted_int)
    elif sample["target_type"] == "boolean":
        boolean_pred = outputs["boolean"]
        predicted_bool = torch.argmax(boolean_pred, dim=1).item()
        print("Predicted boolean:", "yes" if predicted_bool == 1 else "no")
    elif sample["target_type"] == "size":
        size_pred = outputs["size"]
        predicted_size = torch.argmax(size_pred, dim=1).item()
        print("Predicted size token:", vocab.get(str(predicted_size), predicted_size))
    elif sample["target_type"] == "color":
        color_pred = outputs["color"]
        predicted_color = torch.argmax(color_pred, dim=1).item()
        print("Predicted color token:", vocab.get(str(predicted_color), predicted_color))
    elif sample["target_type"] == "shape":
        shape_pred = outputs["shape"]
        predicted_shape = torch.argmax(shape_pred, dim=1).item()
        print("Predicted shape token:", vocab.get(str(predicted_shape), predicted_shape))
    elif sample["target_type"] == "material":
        material_pred = outputs["material"]
        predicted_material = torch.argmax(material_pred, dim=1).item()
        print("Predicted material token:", vocab.get(str(predicted_material), predicted_material))
    elif sample["target_type"] == "vocab":
        vocab_pred = outputs["vocab"]
        predicted_token = torch.argmax(vocab_pred, dim=1).item()
        # Look up the token in the vocabulary.
        token_str = [k for k, v in vocab.items() if v == predicted_token]
        print("Predicted vocab token:", token_str[0] if token_str else predicted_token)
    print("-----")
    
def main():
    questions_file = "/Users/guoyuzhang/University/Y5/diss/vqa/code/flattened_questions.h5"
    features_file = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"
    vocab_file = "/Users/guoyuzhang/University/Y5/diss/vqa/code/vocab.json"

    num_epochs = 5
    batch_size = 16
    learning_rate = 1e-3
    max_seq_len = 20
    embed_dim = 128
    hidden_dim = 256
    max_bbox_steps = 10

    # Use the MPS device if available on Mac, else CPU/GPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Set subset_frac < 1.0 (e.g., 0.001 for quick debugging).
    subset_frac = 0.001

    # Create the dataset.
    dataset = CLEVRDataset(questions_file, features_file, vocab_file, max_seq_len=max_seq_len, subset_frac=subset_frac)
    
    # ===============================
    # Debug: Print out a few training samples.
    # ===============================
    print("Debug: Printing first few training samples with shapes:")
    num_debug_samples = 30  # Change this number if you want more or fewer samples.
    for idx in range(min(num_debug_samples, len(dataset))):
        sample = dataset[idx]
        print(f"Sample {idx}:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}")
            else:
                print(f"  {key}: {value}")
        print("-----")
    # Exit early for debugging; remove or comment out the next line to proceed with training.
    # return
    # ===============================
    # End Debug Block
    # ===============================

    # (If debugging is finished, continue to setup dataloader, model, training, and inference.)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=clevr_collate_fn)

    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    vocab_size = max(vocab.values()) + 1

    model = MultiHeadModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                           image_feat_dim=1024, max_bbox_steps=max_bbox_steps)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, dataloader, optimizer, num_epochs, device)

    # Save the model.
    torch.save(model.state_dict(), "multihead_model.pth")
    print("Model saved to multihead_model.pth")

    # Run inference on a few samples.
    print("\nInference Examples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        run_inference(model, sample, device, vocab)

if __name__ == "__main__":
    main()
