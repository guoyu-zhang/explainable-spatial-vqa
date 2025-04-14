import json
import re
import logging
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Configuration ---
ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"
TRAIN_FEATURES_H5 = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"
BATCH_SIZE = 32
MAX_INPUT_BOXES = 18
FUNCTION_VOCAB_SIZE = 40
FUNCTION_EMB_DIM = 32
TOKEN_VOCAB_SIZE = 29
MODEL_PATH = "token_predictor_best.pth"  # or "token_predictor_best.pth"
MIN_SAMPLES_PER_FUNCTION = 100
SUBSET_FRACTION = 0.01  # Use only 10% of the data; adjust as needed

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Functions ---
def parse_bboxes(bbox_str: str):
    """Parses a string containing bounding box coordinates."""
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
    """Pads a list of bounding boxes to a fixed size."""
    num = len(bboxes)
    mask = [1] * min(num, max_boxes) + [0] * (max_boxes - min(num, max_boxes))
    padded = bboxes[:max_boxes] + [[0.0] * 4] * (max_boxes - len(bboxes))
    return padded, mask

# --- Dataset for Token Prediction ---
class TokenDataset(Dataset):
    """
    A dataset that loads only token prediction samples.
    A sample is considered a token sample if its "output_values" string does not contain '['.
    """
    def __init__(self, annotated_h5_path, features_h5_path, subset_fraction=1.0):
        self.annotated_h5_path = annotated_h5_path
        self.features_h5_path = features_h5_path

        with h5py.File(annotated_h5_path, 'r') as hf:
            self.question_keys = list(hf.keys())

        # Build an index mapping: sample index -> (question_key, step_index, image_index)
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
                for step_idx, step in enumerate(annotated_program):
                    raw_out = step.get("output_values", "").strip()
                    if '[' in raw_out:
                        continue
                    self.samples_index.append((key, step_idx, image_index))
        logging.info(f"Total token samples before subset: {len(self.samples_index)}")
        
        # Apply subset fraction.
        if subset_fraction < 1.0:
            subset_size = int(len(self.samples_index) * subset_fraction)
            self.samples_index = self.samples_index[:subset_size]
            logging.info(f"Using subset fraction {subset_fraction}: {len(self.samples_index)} samples")

        # Lazy loading for features and caching for questions.
        self.features_h5 = None
        self.questions_cache = {}

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        key, step_idx, image_index = self.samples_index[idx]

        # Lazy load question from HDF5 or cache it.
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
        in_vals = step.get("input_values", "").strip()
        raw_out = step.get("output_values", "").strip()
        in_bboxes = parse_bboxes(in_vals)
        in_bboxes_padded, _ = pad_bboxes(in_bboxes, MAX_INPUT_BOXES)
        sample = {
            "image_index": image_index,
            "function": int(step.get("function", "0")),
            "input_bboxes": np.array(in_bboxes_padded, dtype=np.float32),
            "raw_output": raw_out
        }

        # Lazy-load image features.
        if self.features_h5 is None:
            self.features_h5 = h5py.File(self.features_h5_path, 'r')
            self.features = self.features_h5['features']
        img_idx = image_index
        image_feat = torch.tensor(self.features[img_idx], dtype=torch.float32)
        func_tensor = torch.tensor(sample["function"], dtype=torch.long)
        input_bboxes_tensor = torch.tensor(sample["input_bboxes"], dtype=torch.float32)
        try:
            target_token = int(raw_out.strip()) if raw_out.strip() != "" else 0
        except:
            target_token = 0

        return image_feat, func_tensor, input_bboxes_tensor, target_token

# --- Token Predictor Model ---
class TokenPredictor(nn.Module):
    def __init__(self, 
                 function_vocab_size=FUNCTION_VOCAB_SIZE, 
                 function_emb_dim=FUNCTION_EMB_DIM, 
                 max_input_boxes=MAX_INPUT_BOXES, 
                 token_vocab_size=TOKEN_VOCAB_SIZE):
        super(TokenPredictor, self).__init__()
        # Image branch.
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_fc = nn.Linear(1024, 256)
        # Function branch.
        self.func_emb = nn.Embedding(function_vocab_size, function_emb_dim)
        self.func_fc = nn.Linear(function_emb_dim, 32)
        # Bounding box branch.
        self.bbox_fc = nn.Sequential(
            nn.Linear(max_input_boxes * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        # Fused features: 256 + 32 + 64 = 352.
        self.fused_dim = 256 + 32 + 64
        # Token head.
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
        
        x = torch.cat([x_img, x_func, x_bbox], dim=1)
        token_logits = self.token_head(x)
        return token_logits

def run_inference():
    # Select device.
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize dataset and dataloader with a subset fraction.
    dataset = TokenDataset(ANNOTATED_QUESTIONS_H5, TRAIN_FEATURES_H5, subset_fraction=SUBSET_FRACTION)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model and load weights.
    model = TokenPredictor().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info(f"Loaded model from {MODEL_PATH}")

    # Initialize counters for per function accuracy.
    function_counts = [0] * FUNCTION_VOCAB_SIZE
    function_correct = [0] * FUNCTION_VOCAB_SIZE

    with torch.no_grad():
        for image_feat, func_token, input_bboxes, target_token in dataloader:
            image_feat = image_feat.to(device)
            func_token = func_token.to(device)
            input_bboxes = input_bboxes.to(device)
            target_token = target_token.to(device)

            logits = model(image_feat, func_token, input_bboxes)
            predictions = logits.argmax(dim=1)
            
            batch_size = image_feat.size(0)
            for i in range(batch_size):
                func_idx = func_token[i].item()
                function_counts[func_idx] += 1
                if predictions[i].item() == target_token[i].item():
                    function_correct[func_idx] += 1
            
            # Optional: break early if every function has reached MIN_SAMPLES_PER_FUNCTION samples.
            # all_enough = all(count >= MIN_SAMPLES_PER_FUNCTION or count == 0 for count in function_counts)
            # if all_enough:
            #     break

    # Print per function accuracy.
    logging.info("Per Function Accuracy:")
    for func in range(FUNCTION_VOCAB_SIZE):
        if function_counts[func] >= MIN_SAMPLES_PER_FUNCTION:
            acc = function_correct[func] / function_counts[func]
            logging.info(f"Function {func}: Accuracy: {acc * 100:.2f}% ({function_correct[func]}/{function_counts[func]})")
        elif function_counts[func] > 0:
            logging.info(f"Function {func}: Not enough samples (only {function_counts[func]} samples)")
        else:
            logging.info(f"Function {func}: No samples encountered.")

if __name__ == "__main__":
    run_inference()
