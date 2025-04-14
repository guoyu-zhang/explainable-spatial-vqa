import json
import re
from typing import List, Dict, Any
from collections import defaultdict
import logging
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"
TRAIN_FEATURES_H5 = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"

MAX_INPUT_BOXES = 18      # Maximum number of input bounding boxes per sample
MAX_OUTPUT_BOXES = 10     # Maximum number of output bounding boxes per sample
BBOX_DIM = 4              # Each bounding box is represented by 4 numbers

# For function tokens
FUNCTION_VOCAB_SIZE = 40  # Adjust based on the number of unique function tokens
FUNCTION_EMB_DIM = 32

# For token predictions (target is a single integer)
TOKEN_VOCAB_SIZE = 29     # Adjust as needed

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

SUBSET_FRACTION = 0.1

# Loss weights
CONF_LOSS_WEIGHT = 1.0
IOU_LOSS_WEIGHT = 1.0
BRANCH_LOSS_WEIGHT = 1.0  # Weight for branch classification loss

CHECKPOINT_INTERVAL = 10

# --- Helper functions to parse bounding box strings ---
def parse_bboxes(bbox_str: str):
    if not bbox_str or bbox_str.strip() == "":
        return []
    boxes = []
    matches = re.findall(r'\[([^\]]+)\]', bbox_str)
    for match in matches:
        numbers = [float(x) for x in match.strip().split()]
        if len(numbers) == 4:
            boxes.append(numbers)
    return boxes

def pad_bboxes(bboxes, max_boxes):
    num = len(bboxes)
    mask = [1] * min(num, max_boxes) + [0] * (max_boxes - min(num, max_boxes))
    padded = bboxes[:max_boxes] + [[0.0]*4] * (max_boxes - len(bboxes))
    return padded, mask

# --- IoU Loss Function ---
def compute_iou_loss(pred_boxes, target_boxes, mask, eps=1e-6):
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_boxes.unbind(dim=-1)
    target_xmin, target_ymin, target_xmax, target_ymax = target_boxes.unbind(dim=-1)
    
    inter_xmin = torch.max(pred_xmin, target_xmin)
    inter_ymin = torch.max(pred_ymin, target_ymin)
    inter_xmax = torch.min(pred_xmax, target_xmax)
    inter_ymax = torch.min(pred_ymax, target_ymax)
    
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h
    
    area_pred = (pred_xmax - pred_xmin).clamp(min=0) * (pred_ymax - pred_ymin).clamp(min=0)
    area_target = (target_xmax - target_xmin).clamp(min=0) * (target_ymax - target_ymin).clamp(min=0)
    union_area = area_pred + area_target - inter_area + eps
    
    iou = inter_area / union_area
    iou_loss = (1 - iou) * mask
    return iou_loss.sum() / (mask.sum() + eps)

# --- Custom Dataset ---
class BBoxDataset(Dataset):
    def __init__(self, annotated_h5_path, features_h5_path, subset_fraction=1.0):
        self.features_h5_path = features_h5_path
        with h5py.File(annotated_h5_path, 'r') as hf:
            data = hf['questions'][()]
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            annotated_questions = json.loads(data)
        self.features_h5 = None
        self.samples = []
        for q in annotated_questions:
            image_index = q.get("image_index")
            for step in q.get("annotated_program", []):
                raw_out = step.get("output_values", "").strip()
                func = step.get("function", "")
                try:
                    func_int = int(func)
                except ValueError:
                    continue
                in_vals = step.get("input_values", "").strip()
                if '[' in raw_out:
                    in_bboxes = parse_bboxes(in_vals)
                    out_bboxes = parse_bboxes(raw_out)
                    in_bboxes_padded, _ = pad_bboxes(in_bboxes, MAX_INPUT_BOXES)
                    out_bboxes_padded, out_mask = pad_bboxes(out_bboxes, MAX_OUTPUT_BOXES)
                    sample = {
                        "image_index": image_index,
                        "function": func_int,
                        "input_bboxes": np.array(in_bboxes_padded, dtype=np.float32),
                        "target_bboxes": np.array(out_bboxes_padded, dtype=np.float32),
                        "bbox_mask": np.array(out_mask, dtype=np.float32),
                        "raw_output": raw_out  # raw string contains '[' -> bbox sample
                    }
                else:
                    # print(raw_out)
                    dummy_bboxes = [[0.0]*4]*MAX_OUTPUT_BOXES
                    dummy_mask = [0]*MAX_OUTPUT_BOXES
                    sample = {
                        "image_index": image_index,
                        "function": func_int,
                        "input_bboxes": np.array(pad_bboxes(parse_bboxes(in_vals), MAX_INPUT_BOXES)[0], dtype=np.float32),
                        "target_bboxes": np.array(dummy_bboxes, dtype=np.float32),
                        "bbox_mask": np.array(dummy_mask, dtype=np.float32),
                        "raw_output": raw_out  # token sample: no '[' present
                    }
                self.samples.append(sample)
        logging.info(f"Total samples before subsetting: {len(self.samples)}")
        if subset_fraction < 1.0:
            subset_size = int(len(self.samples) * subset_fraction)
            self.samples = self.samples[:subset_size]
            logging.info(f"Using subset_fraction={subset_fraction}, total samples after subsetting: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.features_h5 is None:
            self.features_h5 = h5py.File(self.features_h5_path, 'r')
            self.features = self.features_h5['features']
        img_idx = sample["image_index"]
        image_feat = torch.tensor(self.features[img_idx], dtype=torch.float32)  # (1024, 14, 14)
        func = torch.tensor(sample["function"], dtype=torch.long)
        input_bboxes = torch.tensor(sample["input_bboxes"], dtype=torch.float32)  # (MAX_INPUT_BOXES, 4)
        target_bboxes = torch.tensor(sample["target_bboxes"], dtype=torch.float32)  # (MAX_OUTPUT_BOXES, 4)
        bbox_mask = torch.tensor(sample["bbox_mask"], dtype=torch.float32)  # (MAX_OUTPUT_BOXES,)
        raw_output = sample["raw_output"]
        return image_feat, func, input_bboxes, target_bboxes, bbox_mask, raw_output

# --- Modified Model Definition with Branch Head ---
class MultiTaskBBoxPredictor(nn.Module):
    def __init__(self, function_vocab_size, function_emb_dim, max_input_boxes, max_output_boxes, token_vocab_size):
        super(MultiTaskBBoxPredictor, self).__init__()
        self.max_input_boxes = max_input_boxes
        self.max_output_boxes = max_output_boxes
        # Image branch
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_fc = nn.Linear(1024, 256)
        # Function branch
        self.func_emb = nn.Embedding(function_vocab_size, function_emb_dim)
        self.func_fc = nn.Linear(function_emb_dim, 32)
        # Input bounding boxes branch
        self.bbox_fc = nn.Sequential(
            nn.Linear(max_input_boxes * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.fused_dim = 256 + 32 + 64
        # Branch head: predicts branch (0 for bbox, 1 for token)
        self.branch_head = nn.Linear(self.fused_dim, 2)
        # BBox head: outputs MAX_OUTPUT_BOXES * 5 (4 coords + 1 confidence)
        self.bbox_head = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_output_boxes * 5)
        )
        # Token head: classification over token vocabulary
        self.token_head = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, token_vocab_size)
        )
    
    def forward(self, image_feat, func_token, input_bboxes):
        x_img = self.avg_pool(image_feat)
        x_img = x_img.view(x_img.size(0), -1)
        x_img = self.img_fc(x_img)
        x_func = self.func_emb(func_token)
        x_func = self.func_fc(x_func)
        x_bbox = input_bboxes.view(input_bboxes.size(0), -1)
        x_bbox = self.bbox_fc(x_bbox)
        # Fuse features
        x = torch.cat([x_img, x_func, x_bbox], dim=1)  # (B, fused_dim)
        # Compute branch logits
        branch_logits = self.branch_head(x)  # (B, 2)
        # Compute both branch outputs
        bbox_logits = self.bbox_head(x)
        bbox_out = bbox_logits.view(-1, self.max_output_boxes, 5)
        token_logits = self.token_head(x)  # (B, token_vocab_size)
        return branch_logits, bbox_out, token_logits

# --- Modified Training Loop ---
def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    dataset = BBoxDataset(ANNOTATED_QUESTIONS_H5, TRAIN_FEATURES_H5, subset_fraction=SUBSET_FRACTION)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = MultiTaskBBoxPredictor(FUNCTION_VOCAB_SIZE, FUNCTION_EMB_DIM, MAX_INPUT_BOXES, MAX_OUTPUT_BOXES, TOKEN_VOCAB_SIZE)
    model.to(device)
    
    coord_criterion = nn.MSELoss(reduction='none')
    conf_criterion = nn.BCEWithLogitsLoss(reduction='none')
    token_criterion = nn.CrossEntropyLoss()
    branch_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        steps = 0
        for img_feat, func_token, in_bboxes, target_bboxes, bbox_mask, raw_output in dataloader:
            img_feat = img_feat.to(device)
            func_token = func_token.to(device)
            in_bboxes = in_bboxes.to(device)
            target_bboxes = target_bboxes.to(device)
            bbox_mask = bbox_mask.to(device)
            
            optimizer.zero_grad()
            branch_logits, bbox_out, token_out = model(img_feat, func_token, in_bboxes)
            
            # Create branch labels for each sample: 0 if raw_output contains '[', else 1.
            branch_labels = [0 if '[' in s else 1 for s in raw_output]
            branch_labels = torch.tensor(branch_labels, dtype=torch.long, device=device)
            loss_branch = branch_criterion(branch_logits, branch_labels)
            
            # For bbox samples (branch label 0), compute bbox loss.
            loss_bbox = torch.tensor(0.0, device=device)
            loss_token = torch.tensor(0.0, device=device)
            if branch_labels.eq(0).sum() > 0:
                idx_bbox = (branch_labels == 0).nonzero(as_tuple=False).squeeze(1)
                img_bbox = img_feat[idx_bbox]
                func_bbox = func_token[idx_bbox]
                in_bbox = in_bboxes[idx_bbox]
                target_bbox = target_bboxes[idx_bbox]
                mask_bbox = bbox_mask[idx_bbox]
                # Re-run model for these samples to get outputs (or index from current outputs)
                # Here we index the outputs:
                bbox_out_sel = bbox_out[idx_bbox]
                pred_coords = bbox_out_sel[:, :, :4]
                pred_conf = bbox_out_sel[:, :, 4]
                loss_coords = coord_criterion(pred_coords, target_bbox).mean(dim=2)
                loss_coords = (loss_coords * mask_bbox).sum() / (mask_bbox.sum() + 1e-8)
                loss_iou = compute_iou_loss(pred_coords, target_bbox, mask_bbox)
                loss_bbox_total = loss_coords + IOU_LOSS_WEIGHT * loss_iou
                loss_conf = conf_criterion(pred_conf, mask_bbox).mean()
                loss_bbox = loss_bbox_total + CONF_LOSS_WEIGHT * loss_conf
            
            # For token samples (branch label 1), compute token loss.
            if branch_labels.eq(1).sum() > 0:
                idx_token = (branch_labels == 1).nonzero(as_tuple=False).squeeze(1)
                token_out_sel = token_out[idx_token]  # (N, TOKEN_VOCAB_SIZE)
                # Convert raw_output for token samples to integers.
                target_tokens = [int(raw_output[i].strip()) if raw_output[i].strip() != "" else 0 for i in idx_token.tolist()]
                target_tokens = torch.tensor(target_tokens, dtype=torch.long, device=device)
                loss_token = token_criterion(token_out_sel, target_tokens)
                # print("Raw outputs for token samples:", [raw_output[i] for i in idx_token.tolist()])
                # target_tokens = [int(raw_output[i].strip()) if raw_output[i].strip() != "" else 0 for i in idx_token.tolist()]
                # print("Converted target tokens:", target_tokens)
                # target_tokens = torch.tensor(target_tokens, dtype=torch.long, device=device)
                # loss_token = token_criterion(token_out_sel, target_tokens)
            
            loss = loss_branch + loss_bbox + loss_token
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += img_feat.size(0)
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {total_loss/steps:.4f}")
        
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"multi_task_bbox_predictor_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
            
    logging.info("Training complete.")
    final_model_path = "multi_task_bbox_predictor_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    model.eval()
    inference_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    logging.info("Inference examples:")
    model.eval()
    inference_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    logging.info("Inference examples (only token samples):")
    token_count = 0
    with torch.no_grad():
        for i, (img_feat, func_token, in_bboxes, target_bboxes, bbox_mask, raw_output) in enumerate(inference_loader):
            img_feat = img_feat.to(device)
            func_token = func_token.to(device)
            in_bboxes = in_bboxes.to(device)
            branch_logits, bbox_out, token_out = model(img_feat, func_token, in_bboxes)
            # Get branch prediction (0: bbox, 1: token)
            branch_pred = branch_logits.argmax(dim=1)  # (B,)
            if branch_pred.item() != 1:
                continue  # Skip bbox examples
            token_pred = token_out.argmax(dim=1).cpu().numpy()
            try:
                target_token = int(raw_output[0].strip())
            except:
                target_token = 0
            logging.info(f"Inference Sample (Token sample) {token_count+1}")
            logging.info(f"Predicted token: {token_pred}")
            logging.info(f"Target token: {target_token}")
            token_count += 1
            if token_count >= 40:
                break



if __name__ == "__main__":
    train()
