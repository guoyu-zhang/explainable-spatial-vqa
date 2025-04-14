import json
import re
import logging
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Configuration ---
ANNOTATED_QUESTIONS_H5 = "/Users/guoyuzhang/University/Y5/diss/vqa/code/annotated_questions.h5"
TRAIN_FEATURES_H5 = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"

MAX_INPUT_BOXES = 18      # Maximum number of input bounding boxes per sample
FUNCTION_VOCAB_SIZE = 40  # Based on the number of unique function tokens
FUNCTION_EMB_DIM = 32

# New constant: Use only 10% of the data for inference
INFERENCE_SUBSET_FRACTION = 0.01

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper functions ---
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
    padded = bboxes[:max_boxes] + [[0.0]*4] * (max_boxes - len(bboxes))
    mask = [1] * min(num, max_boxes) + [0] * (max_boxes - min(num, max_boxes))
    return padded, mask

# --- Custom Dataset for Box Selection ---
class BBoxSelectionDataset(Dataset):
    def __init__(self, annotated_h5_path, features_h5_path, subset_fraction=1.0):
        self.annotated_h5_path = annotated_h5_path
        self.features_h5_path = features_h5_path
        self.questions_cache = {}
        self.samples_index = []
        self.ignore_functions = {'12', '36', '23', '26', '31', '17', '19', '30'}
        
        with h5py.File(annotated_h5_path, 'r') as hf:
            self.question_keys = list(hf.keys())
        if subset_fraction < 1.0:
            subset_size = int(len(self.question_keys) * subset_fraction)
            self.question_keys = self.question_keys[:subset_size]
        
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
                for step_idx, step in enumerate(annotated_program):
                    func = step.get("function", "")
                    if func in self.ignore_functions:
                        continue
                    raw_out = step.get("output_values", "").strip()
                    if '[' not in raw_out:
                        continue
                    self.samples_index.append((key, step_idx, image_index))
        logging.info(f"Total samples created: {len(self.samples_index)}")
        self.features_h5 = None

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        key, step_idx, image_index = self.samples_index[idx]
        if key in self.questions_cache:
            question = self.questions_cache[key]
        else:
            with h5py.File(self.annotated_h5_path, 'r') as hf:
                q_str = hf[key][()]
                if isinstance(q_str, bytes):
                    q_str = q_str.decode('utf-8')
                question = json.loads(q_str)
            self.questions_cache[key] = question

        step = question["annotated_program"][step_idx]
        raw_out = step.get("output_values", "").strip()
        in_vals = step.get("input_values", "").strip()
        func = step.get("function", "")
        try:
            func_int = int(func)
        except ValueError:
            func_int = 0

        in_bboxes = parse_bboxes(in_vals)
        out_bboxes = parse_bboxes(raw_out)
        in_bboxes_padded, in_mask = pad_bboxes(in_bboxes, MAX_INPUT_BOXES)
        target_selection = []
        for box in in_bboxes_padded:
            if any(np.allclose(box, obox, atol=1e-6) for obox in out_bboxes):
                target_selection.append(1)
            else:
                target_selection.append(0)
        target_selection = np.array(target_selection, dtype=np.float32)

        if self.features_h5 is None:
            self.features_h5 = h5py.File(self.features_h5_path, 'r')
            self.features = self.features_h5['features']
        image_feat = torch.tensor(self.features[image_index], dtype=torch.float32)
        func_tensor = torch.tensor(func_int, dtype=torch.long)
        input_bboxes_tensor = torch.tensor(in_bboxes_padded, dtype=torch.float32)
        target_selection_tensor = torch.tensor(target_selection, dtype=torch.float32)

        return image_feat, func_tensor, input_bboxes_tensor, target_selection_tensor, raw_out

# --- Model Definition ---
class BBoxSelectionPredictor(nn.Module):
    def __init__(self, function_vocab_size, function_emb_dim, max_input_boxes):
        super(BBoxSelectionPredictor, self).__init__()
        self.max_input_boxes = max_input_boxes
        # Global image branch
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_fc = nn.Linear(1024, 128)
        # Function branch
        self.func_emb = nn.Embedding(function_vocab_size, function_emb_dim)
        self.func_fc = nn.Linear(function_emb_dim, 32)
        # Input bounding boxes branch
        self.box_fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        # Fusion and output head
        self.head = nn.Sequential(
            nn.Linear(176, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, image_feat, func_token, input_bboxes):
        x_img = self.avg_pool(image_feat)  # (B, 1024, 1, 1)
        x_img = x_img.view(x_img.size(0), -1)  # (B, 1024)
        x_img = self.img_fc(x_img)  # (B, 128)
        x_func = self.func_emb(func_token)  # (B, function_emb_dim)
        x_func = self.func_fc(x_func)         # (B, 32)
        global_feat = torch.cat([x_img, x_func], dim=1)  # (B, 160)
        box_feat = self.box_fc(input_bboxes)  # (B, max_input_boxes, 16)
        global_feat_exp = global_feat.unsqueeze(1).expand(-1, self.max_input_boxes, -1)
        combined = torch.cat([global_feat_exp, box_feat], dim=2)  # (B, max_input_boxes, 176)
        logits = self.head(combined)  # (B, max_input_boxes, 1)
        logits = logits.squeeze(-1)   # (B, max_input_boxes)
        return logits

# --- Evaluation Function ---
def evaluate(model, dataset, device):
    model.eval()
    overall_correct = 0
    overall_total = 0
    per_function = {}  # function id -> [correct, total]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for img_feat, func_token, in_bboxes, target_sel, raw_output in loader:
            img_feat = img_feat.to(device)
            func_token = func_token.to(device)
            in_bboxes = in_bboxes.to(device)
            target_sel = target_sel.to(device)
            
            logits = model(img_feat, func_token, in_bboxes)
            predictions = torch.sigmoid(logits).round()  # binary prediction
            correct = (predictions.cpu() == target_sel.cpu()).sum().item()
            total = target_sel.numel()
            
            overall_correct += correct
            overall_total += total
            
            func_id = int(func_token.item())
            if func_id not in per_function:
                per_function[func_id] = [0, 0]
            per_function[func_id][0] += correct
            per_function[func_id][1] += total
            
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    per_function_accuracy = {func: (stats[0] / stats[1] if stats[1] > 0 else 0.0)
                             for func, stats in per_function.items()}
    return overall_accuracy, per_function_accuracy

# --- Inference Main Routine ---
def main():
    # Set device (use CUDA or fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    logging.info(f"Using device: {device}")
    
    # Create dataset using only 10% of the data for inference.
    dataset = BBoxSelectionDataset(ANNOTATED_QUESTIONS_H5, TRAIN_FEATURES_H5, subset_fraction=INFERENCE_SUBSET_FRACTION)
    
    # Instantiate the model and load the saved checkpoint
    model = BBoxSelectionPredictor(FUNCTION_VOCAB_SIZE, FUNCTION_EMB_DIM, MAX_INPUT_BOXES)
    model.to(device)
    
    checkpoint_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/bbox_selection_predictor_best.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Loaded model checkpoint from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return
    
    # --- Run Evaluation ---
    overall_acc, per_function_acc = evaluate(model, dataset, device)
    logging.info(f"Overall Accuracy: {overall_acc:.4f}")
    logging.info("Per Function Accuracy:")
    for func, acc in per_function_acc.items():
        logging.info(f"  Function {func}: {acc:.4f}")
    
    # --- Run and Print a Few Inference Examples ---
    inference_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    sample_count = 0
    logging.info("Inference Samples:")
    with torch.no_grad():
        for img_feat, func_token, in_bboxes, target_sel, raw_output in inference_loader:
            img_feat = img_feat.to(device)
            func_token = func_token.to(device)
            in_bboxes = in_bboxes.to(device)
            logits = model(img_feat, func_token, in_bboxes)
            pred = torch.sigmoid(logits).round().cpu().numpy()  # binary predictions per box
            logging.info(f"Sample {sample_count+1}:")
            logging.info(f"  Predicted selection: {pred}")
            logging.info(f"  Target selection: {target_sel.numpy()}")
            sample_count += 1
            if sample_count >= 5:
                break

if __name__ == "__main__":
    main()
