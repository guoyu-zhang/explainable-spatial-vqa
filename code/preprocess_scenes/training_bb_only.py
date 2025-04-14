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
TRAIN_FEATURES_H5 = "/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5"

MAX_INPUT_BOXES = 18      # Maximum number of input bounding boxes per sample
MAX_OUTPUT_BOXES = 10     # Maximum number of output bounding boxes per sample
BBOX_DIM = 4              # Each bounding box is represented by 4 numbers

# For function tokens
FUNCTION_VOCAB_SIZE = 40  # Adjust based on the number of unique function tokens
FUNCTION_EMB_DIM = 32

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# Set the fraction of data to use for training
SUBSET_FRACTION = 0.01

# Weight for the confidence loss (you may adjust this)
CONF_LOSS_WEIGHT = 1.0

# Checkpoint saving interval (in epochs)
CHECKPOINT_INTERVAL = 10

# --- Helper functions to parse bounding box strings ---
def parse_bboxes(bbox_str: str):
    """
    Parses a string of bounding boxes.
    Example: "[0.4939 0.1753 0.6269 0.3747] [0.1240 0.1913 0.2635 0.4964]"
    returns a list of lists:
      [[0.4939, 0.1753, 0.6269, 0.3747], [0.1240, 0.1913, 0.2635, 0.4964]]
    """
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
    """
    Pads a list of bounding boxes (each a list of 4 floats) with zeros up to max_boxes.
    Also returns a mask of shape (max_boxes,), with 1 for valid boxes and 0 for padding.
    """
    num = len(bboxes)
    mask = [1] * min(num, max_boxes) + [0] * (max_boxes - min(num, max_boxes))
    padded = bboxes[:max_boxes] + [[0.0]*4] * (max_boxes - len(bboxes))
    return padded, mask

# --- Custom Dataset ---
class BBoxDataset(Dataset):
    def __init__(self, annotated_h5_path, features_h5_path, subset_fraction=1.0):
        self.features_h5_path = features_h5_path
        # Load annotated questions (JSON list) from HDF5 file.
        with h5py.File(annotated_h5_path, 'r') as hf:
            data = hf['questions'][()]
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            annotated_questions = json.loads(data)
        
        # We don't open the features file here; we'll do it lazily in __getitem__
        self.features_h5 = None
        
        self.samples = []
        for q in annotated_questions:
            image_index = q.get("image_index")
            for step in q.get("annotated_program", []):
                out_val = step.get("output_values", "")
                # Only keep steps where output_values contain bounding boxes.
                if '[' not in out_val:
                    continue
                func = step.get("function", "")
                try:
                    func_int = int(func)
                except ValueError:
                    continue
                in_bboxes = parse_bboxes(step.get("input_values", ""))
                out_bboxes = parse_bboxes(out_val)
                # Pad input boxes to MAX_INPUT_BOXES and output boxes to MAX_OUTPUT_BOXES.
                in_bboxes_padded, _ = pad_bboxes(in_bboxes, MAX_INPUT_BOXES)
                out_bboxes_padded, out_mask = pad_bboxes(out_bboxes, MAX_OUTPUT_BOXES)
                sample = {
                    "image_index": image_index,
                    "function": func_int,
                    "input_bboxes": np.array(in_bboxes_padded, dtype=np.float32),   # (MAX_INPUT_BOXES, 4)
                    "target_bboxes": np.array(out_bboxes_padded, dtype=np.float32),   # (MAX_OUTPUT_BOXES, 4)
                    "bbox_mask": np.array(out_mask, dtype=np.float32)  # mask for output boxes
                }
                self.samples.append(sample)
        logging.info(f"Total samples before subsetting: {len(self.samples)}")
        # Keep only a fraction of the samples
        if subset_fraction < 1.0:
            subset_size = int(len(self.samples) * subset_fraction)
            self.samples = self.samples[:subset_size]
            logging.info(f"Using subset_fraction={subset_fraction}, total samples after subsetting: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Lazy-load the features HDF5 file in each worker if not already opened.
        if self.features_h5 is None:
            self.features_h5 = h5py.File(self.features_h5_path, 'r')
            self.features = self.features_h5['features']
        
        img_idx = sample["image_index"]
        image_feat = self.features[img_idx]  # shape: (1024, 14, 14)
        image_feat = torch.tensor(image_feat, dtype=torch.float32)
        func = torch.tensor(sample["function"], dtype=torch.long)
        input_bboxes = torch.tensor(sample["input_bboxes"], dtype=torch.float32)  # (MAX_INPUT_BOXES, 4)
        target_bboxes = torch.tensor(sample["target_bboxes"], dtype=torch.float32)  # (MAX_OUTPUT_BOXES, 4)
        bbox_mask = torch.tensor(sample["bbox_mask"], dtype=torch.float32)  # (MAX_OUTPUT_BOXES,)
        return image_feat, func, input_bboxes, target_bboxes, bbox_mask

# --- Model Definition ---
class BBoxPredictor(nn.Module):
    def __init__(self, function_vocab_size, function_emb_dim, max_input_boxes, max_output_boxes):
        super(BBoxPredictor, self).__init__()
        self.max_input_boxes = max_input_boxes
        self.max_output_boxes = max_output_boxes
        # Image branch: global average pooling and FC to get image embedding.
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_fc = nn.Linear(1024, 256)
        # Function branch: embedding and FC.
        self.func_emb = nn.Embedding(function_vocab_size, function_emb_dim)
        self.func_fc = nn.Linear(function_emb_dim, 32)
        # Input bounding boxes branch: flatten (max_input_boxes, 4) to (max_input_boxes*4) and process.
        self.bbox_fc = nn.Sequential(
            nn.Linear(max_input_boxes * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        # Fusion: concatenate image, function, and bbox embeddings and output MAX_OUTPUT_BOXES boxes.
        # Now we output 5 values per box (4 coordinates + 1 confidence score)
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 32 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, max_output_boxes * 5)
        )
    
    def forward(self, image_feat, func_token, input_bboxes):
        # Process image features: (B, 1024, 14, 14) -> (B, 1024) -> (B, 256)
        x_img = self.avg_pool(image_feat)
        x_img = x_img.view(x_img.size(0), -1)
        x_img = self.img_fc(x_img)
        
        # Process function token: (B,) -> (B, function_emb_dim) -> (B, 32)
        x_func = self.func_emb(func_token)
        x_func = self.func_fc(x_func)
        
        # Process input bounding boxes: (B, max_input_boxes, 4) -> flatten to (B, max_input_boxes*4) -> (B, 64)
        x_bbox = input_bboxes.view(input_bboxes.size(0), -1)
        x_bbox = self.bbox_fc(x_bbox)
        
        # Concatenate all embeddings.
        x = torch.cat([x_img, x_func, x_bbox], dim=1)
        x = self.fusion_fc(x)
        # Reshape to (B, max_output_boxes, 5)
        out = x.view(-1, self.max_output_boxes, 5)
        return out

# --- Training Loop ---
def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    dataset = BBoxDataset(ANNOTATED_QUESTIONS_H5, TRAIN_FEATURES_H5, subset_fraction=SUBSET_FRACTION)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = BBoxPredictor(FUNCTION_VOCAB_SIZE, FUNCTION_EMB_DIM, MAX_INPUT_BOXES, MAX_OUTPUT_BOXES)
    model.to(device)
    
    # For coordinate regression
    coord_criterion = nn.MSELoss(reduction='none')
    # For confidence prediction, use BCEWithLogitsLoss
    conf_criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        for i, (img_feat, func_token, in_bboxes, target_bboxes, bbox_mask) in enumerate(dataloader):
            img_feat = img_feat.to(device)
            func_token = func_token.to(device)
            in_bboxes = in_bboxes.to(device)
            target_bboxes = target_bboxes.to(device)
            bbox_mask = bbox_mask.to(device)  # (B, MAX_OUTPUT_BOXES)
            
            optimizer.zero_grad()
            outputs = model(img_feat, func_token, in_bboxes)  # (B, MAX_OUTPUT_BOXES, 5)
            # Separate coordinates and confidence predictions
            pred_coords = outputs[:, :, :4]    # (B, MAX_OUTPUT_BOXES, 4)
            pred_conf = outputs[:, :, 4]         # (B, MAX_OUTPUT_BOXES)
            
            # Coordinate loss: only computed on valid boxes
            loss_coords = coord_criterion(pred_coords, target_bboxes)  # (B, MAX_OUTPUT_BOXES, 4)
            loss_coords = loss_coords.mean(dim=2)  # (B, MAX_OUTPUT_BOXES)
            loss_coords = (loss_coords * bbox_mask).sum() / (bbox_mask.sum() + 1e-8)
            
            # Confidence loss: target is bbox_mask (1 for valid boxes, 0 for padded)
            loss_conf = conf_criterion(pred_conf, bbox_mask)  # (B, MAX_OUTPUT_BOXES)
            loss_conf = loss_conf.mean(dim=1).mean()  # average over batch
            
            total_loss = loss_coords + CONF_LOSS_WEIGHT * loss_conf
            total_loss.backward()
            optimizer.step()
            
            logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {total_loss.item():.4f}")
        
        # Save a checkpoint at the end of each epoch (if interval condition is met)
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"bbox_predictor_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
            
    logging.info("Training complete.")
    
    # --- Save the Final Model ---
    final_model_path = "bbox_predictor_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # --- Inference Examples ---
    model.eval()
    inference_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    logging.info("Inference examples:")
    with torch.no_grad():
        for i, (img_feat, func_token, in_bboxes, target_bboxes, bbox_mask) in enumerate(inference_loader):
            img_feat = img_feat.to(device)
            func_token = func_token.to(device)
            in_bboxes = in_bboxes.to(device)
            outputs = model(img_feat, func_token, in_bboxes)  # (1, MAX_OUTPUT_BOXES, 5)
            pred_coords = outputs[:, :, :4].cpu().numpy()
            pred_conf = outputs[:, :, 4].cpu().numpy()
            gt_coords = target_bboxes.cpu().numpy()
            logging.info(f"Inference Sample {i+1}")
            logging.info(f"Predicted bounding boxes:\n{pred_coords}")
            logging.info(f"Predicted confidence scores:\n{pred_conf}")
            logging.info(f"Target bounding boxes:\n{gt_coords}")
            if i >= 4:
                break

if __name__ == "__main__":
    train()
