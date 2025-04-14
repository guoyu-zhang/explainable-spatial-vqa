import json
import re
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
TRAIN_FEATURES_H5 = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"             # Path to your image features HDF5 file
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
SUBSET_FRACTION = 1                                 # Use all token samples; adjust as needed
MAX_INPUT_BOXES = 18                                   # Maximum number of input bounding boxes per sample
FUNCTION_VOCAB_SIZE = 40                               # Adjust based on your dataset (for function tokens)
FUNCTION_EMB_DIM = 32
TOKEN_VOCAB_SIZE = 29                                  # Adjust as needed

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

        # Open the annotated questions HDF5 file to get all question keys.
        with h5py.File(annotated_h5_path, 'r') as hf:
            self.question_keys = list(hf.keys())
        if subset_fraction < 1.0:
            subset_size = int(len(self.question_keys) * subset_fraction)
            self.question_keys = self.question_keys[:subset_size]

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
                # Only include samples that are token predictions (raw_output does not contain '[')
                for step_idx, step in enumerate(annotated_program):
                    raw_out = step.get("output_values", "").strip()
                    if '[' in raw_out:
                        continue
                    self.samples_index.append((key, step_idx, image_index))
        logging.info(f"Total token samples created: {len(self.samples_index)}")

        # Lazy loading for features and caching for questions.
        self.features_h5 = None
        self.questions_cache = {}

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        key, step_idx, image_index = self.samples_index[idx]

        # Lazy load the question from HDF5 or cache it.
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
        # Process input bounding boxes and pad them.
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
        # Convert raw output to an integer token target.
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
        # Image branch: process image features.
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_fc = nn.Linear(1024, 256)
        # Function branch: embed and project function tokens.
        self.func_emb = nn.Embedding(function_vocab_size, function_emb_dim)
        self.func_fc = nn.Linear(function_emb_dim, 32)
        # Bounding box branch: flatten and process input bounding boxes.
        self.bbox_fc = nn.Sequential(
            nn.Linear(max_input_boxes * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        # Fused features: 256 (image) + 32 (function) + 64 (bbox) = 352.
        self.fused_dim = 256 + 32 + 64
        # Token head: classify over the token vocabulary.
        self.token_head = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, token_vocab_size)
        )
        
    def forward(self, image_feat, func_token, input_bboxes):
        # Process image features.
        x_img = self.avg_pool(image_feat)
        x_img = x_img.view(x_img.size(0), -1)
        x_img = self.img_fc(x_img)
        # Process function tokens.
        x_func = self.func_emb(func_token)
        x_func = self.func_fc(x_func)
        # Process input bounding boxes.
        x_bbox = input_bboxes.view(input_bboxes.size(0), -1)
        x_bbox = self.bbox_fc(x_bbox)
        # Fuse the features.
        x = torch.cat([x_img, x_func, x_bbox], dim=1)
        # Predict token logits.
        token_logits = self.token_head(x)
        return token_logits

# --- Training Loop with Logging, Checkpointing, and Inference Evaluation ---
def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    dataset = TokenDataset(ANNOTATED_QUESTIONS_H5, TRAIN_FEATURES_H5, subset_fraction=SUBSET_FRACTION)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = TokenPredictor().to(device)
    token_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    patience = 2  # Number of epochs without improvement allowed
    epochs_since_improvement = 0
    
    loss_log_file = "token_training_losses.txt"
    with open(loss_log_file, "w") as log_f:
        log_f.write("Epoch,Average Loss\n")
    
    # Training loop.
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        steps = 0
        for image_feat, func_token, input_bboxes, target_token in dataloader:
            image_feat = image_feat.to(device)
            func_token = func_token.to(device)
            input_bboxes = input_bboxes.to(device)
            target_token = target_token.to(device)
            
            optimizer.zero_grad()
            token_logits = model(image_feat, func_token, input_bboxes)
            loss = token_criterion(token_logits, target_token)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * image_feat.size(0)
            steps += image_feat.size(0)
        
        avg_epoch_loss = total_loss / steps if steps > 0 else 0
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_epoch_loss:.4f}")
        with open(loss_log_file, "a") as log_f:
            log_f.write(f"{epoch+1},{avg_epoch_loss:.4f}\n")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_since_improvement = 0
            best_model_path = "token_predictor_best.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved to {best_model_path}")
        else:
            epochs_since_improvement += 1
            logging.info(f"No improvement in epoch {epoch+1}. Patience: {epochs_since_improvement}/{patience}")
            if epochs_since_improvement >= patience:
                logging.info("Early stopping triggered.")
                break
        
    final_model_path = "token_predictor_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # --- Inference Evaluation ---
    model.eval()
    total_correct = 0
    total_samples = 0
    token_counts = [0] * TOKEN_VOCAB_SIZE
    token_correct = [0] * TOKEN_VOCAB_SIZE
    # For per function accuracy.
    function_counts = [0] * FUNCTION_VOCAB_SIZE
    function_correct = [0] * FUNCTION_VOCAB_SIZE
    inference_examples_printed = 0  # Counter for printed inference examples

    inference_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    with torch.no_grad():
        for image_feat, func_token, input_bboxes, target_token in inference_loader:
            image_feat = image_feat.to(device)
            func_token = func_token.to(device)
            input_bboxes = input_bboxes.to(device)
            target_token = target_token.to(device)
            logits = model(image_feat, func_token, input_bboxes)
            predictions = logits.argmax(dim=1)
            
            # Update overall and per token accuracy.
            correct = (predictions == target_token).sum().item()
            total_correct += correct
            total_samples += image_feat.size(0)
            for gt, pred in zip(target_token.cpu().numpy(), predictions.cpu().numpy()):
                token_counts[gt] += 1
                if gt == pred:
                    token_correct[gt] += 1
            
            # Update per function accuracy.
            for f, gt, pred in zip(func_token.cpu().numpy(), target_token.cpu().numpy(), predictions.cpu().numpy()):
                function_counts[f] += 1
                if gt == pred:
                    function_correct[f] += 1
            
            # Print up to 40 inference examples.
            for i in range(image_feat.size(0)):
                if inference_examples_printed < 40:
                    logging.info(f"Inference Example {inference_examples_printed + 1}:")
                    logging.info(f"  Function: {func_token[i].item()}, Ground Truth Token: {target_token[i].item()}, Predicted Token: {predictions[i].item()}")
                    inference_examples_printed += 1
                else:
                    break
            if inference_examples_printed >= 40:
                break
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    logging.info(f"Inference Overall Accuracy: {overall_accuracy * 100:.2f}%")
    for token in range(TOKEN_VOCAB_SIZE):
        if token_counts[token] > 0:
            acc = token_correct[token] / token_counts[token]
            logging.info(f"Token {token} Accuracy: {acc * 100:.2f}% ({token_correct[token]}/{token_counts[token]})")
        else:
            logging.info(f"Token {token} Accuracy: No samples.")
    
    for func in range(FUNCTION_VOCAB_SIZE):
        if function_counts[func] > 0:
            acc = function_correct[func] / function_counts[func]
            logging.info(f"Function {func} Accuracy: {acc * 100:.2f}% ({function_correct[func]}/{function_counts[func]})")
        else:
            logging.info(f"Function {func} Accuracy: No samples.")

if __name__ == "__main__":
    train()
