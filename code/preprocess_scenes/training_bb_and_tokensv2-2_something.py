import json
import re
from typing import List, Dict, Any
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

SUBSET_FRACTION = 0.001

# Loss weights for branch loss etc.
BRANCH_LOSS_WEIGHT = 1.0  # Weight for branch classification loss
# Additional multipliers for the bbox loss components (these are fixed multipliers for IoU and confidence)
IOU_LOSS_WEIGHT = 1
CONF_LOSS_WEIGHT = 1

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

# --- Differentiable IoU computation ---
def compute_iou_differentiable(pred_boxes, target_boxes, eps=1e-6):
    # pred_boxes, target_boxes: shape (N, 4)
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_boxes.unbind(dim=1)
    target_xmin, target_ymin, target_xmax, target_ymax = target_boxes.unbind(dim=1)
    
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
    return iou

# --- Sinkhorn Matching Functions ---
def sinkhorn(log_alpha, n_iters=20):
    """
    Performs Sinkhorn normalization on the log_alpha matrix.
    log_alpha: Tensor of shape (num_pred, num_valid), where higher values indicate better matches.
    Returns a doubly-stochastic matrix of the same shape.
    """
    for _ in range(n_iters):
        # Normalize rows
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        # Normalize columns
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return torch.exp(log_alpha)

# --- IoU Loss Function (if needed for entire set) ---
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
# --- Custom Dataset with Lazy Loading of Annotated Questions ---
class BBoxDataset(Dataset):
    def __init__(self, annotated_h5_path, features_h5_path, subset_fraction=1.0):
        self.annotated_h5_path = annotated_h5_path
        self.features_h5_path = features_h5_path

        # Open the HDF5 file to list all keys (each key is a separate question)
        with h5py.File(annotated_h5_path, 'r') as hf:
            self.question_keys = list(hf.keys())
        if subset_fraction < 1.0:
            subset_size = int(len(self.question_keys) * subset_fraction)
            self.question_keys = self.question_keys[:subset_size]

        # Build an index mapping sample index -> (question_key, step_index, image_index)
        self.samples_index = []
        with h5py.File(annotated_h5_path, 'r') as hf:
            for key in self.question_keys:
                q_str = hf[key][()]
                if isinstance(q_str, bytes):
                    q_str = q_str.decode('utf-8')
                try:
                    question = json.loads(q_str)
                except Exception as e:
                    logging.error(f"Error parsing question in {key}: {e}")
                    continue
                image_index = question.get("image_index")
                annotated_program = question.get("annotated_program", [])
                # For each step, add a sample if the step's function can be interpreted as an integer
                for step_idx, step in enumerate(annotated_program):
                    func = step.get("function", "")
                    try:
                        int(func)
                    except ValueError:
                        continue
                    self.samples_index.append((key, step_idx, image_index))
        logging.info(f"Total samples created: {len(self.samples_index)}")

        # Features file is loaded lazily in __getitem__
        self.features_h5 = None
        # Cache to store questions already loaded
        self.questions_cache = {}

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        key, step_idx, image_index = self.samples_index[idx]

        # Lazy load the question from HDF5 (or from cache)
        if key in self.questions_cache:
            question = self.questions_cache[key]
        else:
            with h5py.File(self.annotated_h5_path, 'r') as hf:
                q_str = hf[key][()]
                if isinstance(q_str, bytes):
                    q_str = q_str.decode('utf-8')
                question = json.loads(q_str)
            self.questions_cache[key] = question

        # Process the specific step from the annotated_program
        step = question["annotated_program"][step_idx]
        raw_out = step.get("output_values", "").strip()
        func = step.get("function", "")
        try:
            func_int = int(func)
        except ValueError:
            func_int = 0  # fallback if conversion fails
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
                "raw_output": raw_out
            }
        else:
            dummy_bboxes = [[0.0]*4]*MAX_OUTPUT_BOXES
            dummy_mask = [0]*MAX_OUTPUT_BOXES
            sample = {
                "image_index": image_index,
                "function": func_int,
                "input_bboxes": np.array(pad_bboxes(parse_bboxes(in_vals), MAX_INPUT_BOXES)[0], dtype=np.float32),
                "target_bboxes": np.array(dummy_bboxes, dtype=np.float32),
                "bbox_mask": np.array(dummy_mask, dtype=np.float32),
                "raw_output": raw_out
            }

        # Lazy-load image features
        if self.features_h5 is None:
            self.features_h5 = h5py.File(self.features_h5_path, 'r')
            self.features = self.features_h5['features']
        img_idx = image_index
        image_feat = torch.tensor(self.features[img_idx], dtype=torch.float32)
        func_tensor = torch.tensor(sample["function"], dtype=torch.long)
        input_bboxes_tensor = torch.tensor(sample["input_bboxes"], dtype=torch.float32)
        target_bboxes_tensor = torch.tensor(sample["target_bboxes"], dtype=torch.float32)
        bbox_mask_tensor = torch.tensor(sample["bbox_mask"], dtype=torch.float32)
        raw_output = sample["raw_output"]

        return image_feat, func_tensor, input_bboxes_tensor, target_bboxes_tensor, bbox_mask_tensor, raw_output


# --- Model Definition with Branch Head ---
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

# --- Training Loop with Soft Matching Loss, Padding Handling, and Learnable Cost Weights ---
def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    dataset = BBoxDataset(ANNOTATED_QUESTIONS_H5, TRAIN_FEATURES_H5, subset_fraction=SUBSET_FRACTION)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = MultiTaskBBoxPredictor(FUNCTION_VOCAB_SIZE, FUNCTION_EMB_DIM, MAX_INPUT_BOXES, MAX_OUTPUT_BOXES, TOKEN_VOCAB_SIZE)
    model.to(device)
    
    # Define learnable weights for the cost components.
    cost_weight_l1 = nn.Parameter(torch.tensor(1.0, device=device))
    cost_weight_iou = nn.Parameter(torch.tensor(1.0, device=device))
    
    coord_criterion = nn.MSELoss(reduction='none')
    conf_criterion = nn.BCEWithLogitsLoss(reduction='mean')  # using mean reduction here
    token_criterion = nn.CrossEntropyLoss()
    branch_criterion = nn.CrossEntropyLoss()
    
    # Include cost weights in the optimizer.
    optimizer = optim.Adam(list(model.parameters()) + [cost_weight_l1, cost_weight_iou], lr=LEARNING_RATE)
    
    model.train()
    
    best_loss = float('inf')
    patience = 2  # epochs without improvement allowed
    epochs_since_improvement = 0

    loss_log_file = "training_losses.txt"
    with open(loss_log_file, "w") as log_f:
        log_f.write("Epoch,Average Loss\n")
    
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
            
            # Branch labels: 0 if bbox sample (raw_output contains '['), else 1.
            branch_labels = [0 if '[' in s else 1 for s in raw_output]
            branch_labels = torch.tensor(branch_labels, dtype=torch.long, device=device)
            loss_branch = branch_criterion(branch_logits, branch_labels)
            
            loss_bbox = torch.tensor(0.0, device=device)
            loss_token = torch.tensor(0.0, device=device)
            
            if branch_labels.eq(0).sum() > 0:
                idx_bbox = (branch_labels == 0).nonzero(as_tuple=False).squeeze(1)
                loss_bbox_sum = 0.0
                for sample_idx in idx_bbox:
                    # predicted boxes and confidence (shape: (MAX_OUTPUT_BOXES, ...))
                    pred_boxes = bbox_out[sample_idx, :, :4]      # (M, 4)
                    pred_conf = bbox_out[sample_idx, :, 4]          # (M,)
                    target_boxes_sample = target_bboxes[sample_idx] # (M, 4)
                    mask_sample = bbox_mask[sample_idx]             # (M,)
                    
                    # Select valid targets using mask.
                    valid_idx = (mask_sample == 1).nonzero(as_tuple=True)[0]
                    if valid_idx.numel() == 0:
                        continue
                    valid_target_boxes = target_boxes_sample[valid_idx]  # (num_valid, 4)
                    num_valid = valid_target_boxes.shape[0]
                    num_pred = pred_boxes.shape[0]
                    
                    # Compute cost matrices:
                    pred_boxes_exp = pred_boxes.unsqueeze(1)   # (num_pred, 1, 4)
                    valid_target_boxes_exp = valid_target_boxes.unsqueeze(0)  # (1, num_valid, 4)
                    l1_cost = torch.abs(pred_boxes_exp - valid_target_boxes_exp).sum(dim=2)  # (num_pred, num_valid)
                    
                    iou_cost = torch.zeros_like(l1_cost)
                    for i in range(num_pred):
                        for j in range(num_valid):
                            pb = pred_boxes[i].unsqueeze(0)  # (1,4)
                            tb = valid_target_boxes[j].unsqueeze(0)  # (1,4)
                            iou_val = compute_iou_differentiable(pb, tb)[0]
                            iou_cost[i, j] = 1 - iou_val
                            
                    # Combined cost with learnable weights
                    combined_cost = cost_weight_l1 * l1_cost + cost_weight_iou * iou_cost
                    
                    # Compute soft assignment matrix using Sinkhorn
                    log_alpha = -combined_cost  # higher probability for lower cost
                    P = sinkhorn(log_alpha, n_iters=20)  # shape: (num_pred, num_valid)
                    
                    # Soft coordinate and IoU loss: weighted average over cost matrix
                    loss_coords_sample = (P * l1_cost).sum() / P.sum()
                    loss_iou_sample = (P * iou_cost).sum() / P.sum()
                    
                    # For confidence, we take the max probability for each prediction
                    soft_target_conf = P.max(dim=1)[0]  # shape: (num_pred,)
                    loss_conf_sample = conf_criterion(pred_conf, soft_target_conf)
                    
                    sample_loss = loss_coords_sample + IOU_LOSS_WEIGHT * loss_iou_sample + CONF_LOSS_WEIGHT * loss_conf_sample
                    loss_bbox_sum += sample_loss
                loss_bbox = loss_bbox_sum / idx_bbox.size(0)
            
            if branch_labels.eq(1).sum() > 0:
                idx_token = (branch_labels == 1).nonzero(as_tuple=False).squeeze(1)
                token_out_sel = token_out[idx_token]  # (N, TOKEN_VOCAB_SIZE)
                target_tokens = [int(raw_output[i].strip()) if raw_output[i].strip() != "" else 0 for i in idx_token.tolist()]
                target_tokens = torch.tensor(target_tokens, dtype=torch.long, device=device)
                loss_token = token_criterion(token_out_sel, target_tokens)
            
            loss = loss_branch + loss_bbox + loss_token
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += img_feat.size(0)
        
        avg_epoch_loss = total_loss / steps
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_epoch_loss:.4f}")
        with open(loss_log_file, "a") as log_f:
            log_f.write(f"{epoch+1},{avg_epoch_loss:.4f}\n")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_since_improvement = 0
            best_model_path = "multi_task_bbox_predictor_best.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved to {best_model_path}")
        else:
            epochs_since_improvement += 1
            logging.info(f"No improvement in epoch {epoch+1}. Patience: {epochs_since_improvement}/{patience}")
            if epochs_since_improvement >= patience:
                logging.info("Early stopping triggered.")
                break
        
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"multi_task_bbox_predictor_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
            
    logging.info("Training complete.")
    final_model_path = "multi_task_bbox_predictor_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    model.eval()
    # --- Inference: Token Examples ---
    inference_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    logging.info("Inference examples (Token samples):")
    token_count = 0
    with torch.no_grad():
        for i, (img_feat, func_token, in_bboxes, target_bboxes, bbox_mask, raw_output) in enumerate(inference_loader):
            img_feat = img_feat.to(device)
            func_token = func_token.to(device)
            in_bboxes = in_bboxes.to(device)
            branch_logits, bbox_out, token_out = model(img_feat, func_token, in_bboxes)
            branch_pred = branch_logits.argmax(dim=1)
            if branch_pred.item() != 1:
                continue
            token_pred = token_out.argmax(dim=1).cpu().numpy()
            try:
                target_token = int(raw_output[0].strip())
            except:
                target_token = 0
            logging.info(f"Inference Sample (Token) {token_count+1}")
            logging.info(f"Predicted token: {token_pred}")
            logging.info(f"Target token: {target_token}")
            token_count += 1
            if token_count >= 20:
                break

    # --- Inference: Bounding Box Examples ---
    inference_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    logging.info("Inference examples (Bounding box samples):")
    bbox_count = 0
    with torch.no_grad():
        for i, (img_feat, func_token, in_bboxes, target_bboxes, bbox_mask, raw_output) in enumerate(inference_loader):
            img_feat = img_feat.to(device)
            func_token = func_token.to(device)
            in_bboxes = in_bboxes.to(device)
            branch_logits, bbox_out, token_out = model(img_feat, func_token, in_bboxes)
            branch_pred = branch_logits.argmax(dim=1)
            if branch_pred.item() != 0:
                continue
            predicted_boxes = bbox_out[0].cpu().numpy()  # (MAX_OUTPUT_BOXES, 5)
            target_boxes = target_bboxes[0].cpu().numpy()  # (MAX_OUTPUT_BOXES, 4)
            logging.info(f"Inference Sample (Bounding box) {bbox_count+1}")
            logging.info(f"Predicted boxes (x, y, w, h, conf): {predicted_boxes}")
            logging.info(f"Target boxes: {target_boxes}")
            bbox_count += 1
            if bbox_count >= 20:
                break

if __name__ == "__main__":
    train()
