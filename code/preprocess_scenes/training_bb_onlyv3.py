import os
import json
import re
import h5py
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

##########################
# Helper Functions
##########################

def parse_bboxes(bbox_str):
    """
    Parse a string of bounding boxes.
    Example input:
      "[0.4939 0.1753 0.6269 0.3747] [0.1240 0.1913 0.2635 0.4964] ..."
    Returns a list of boxes, where each box is a list of 4 floats.
    """
    boxes = []
    matches = re.findall(r'\[([^\]]+)\]', bbox_str)
    for match in matches:
        numbers = [float(x) for x in match.strip().split()]
        if len(numbers) == 4:
            boxes.append(numbers)
    return boxes

def convert_to_yolo(box):
    """
    Convert a bounding box from [xmin, ymin, xmax, ymax]
    to [x_center, y_center, width, height]. 
    Assumes that coordinates are normalized (i.e. between 0 and 1).
    """
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return [x_center, y_center, width, height]

##########################
# Dataset Definition
##########################

class YOLODataset(Dataset):
    def __init__(self, annotated_h5_path, images_dir, transform=None, subset_fraction=1.0, S=7):
        """
        annotated_h5_path: Path to the HDF5 file with JSON annotations.
        images_dir: Directory containing the .png images.
        transform: torchvision transforms to apply to the image.
        subset_fraction: Fraction of the dataset to use (for debugging).
        S: Grid size (S x S) for the YOLO target.
        """
        self.annotated_h5_path = annotated_h5_path
        self.images_dir = images_dir
        self.transform = transform
        self.S = S
        self.samples = []
        
        with h5py.File(annotated_h5_path, 'r') as hf:
            keys = list(hf.keys())
            if subset_fraction < 1.0:
                keys = keys[:int(len(keys) * subset_fraction)]
            for key in keys:
                q_str = hf[key][()]
                if isinstance(q_str, bytes):
                    q_str = q_str.decode('utf-8')
                try:
                    question = json.loads(q_str)
                except Exception as e:
                    logging.error(f"Error parsing question {key}: {e}")
                    continue
                
                # Use the image index from the JSON to build the filename.
                image_index = question.get("image_index")
                if image_index is None:
                    continue  # Skip if no image index
                # Format the filename, e.g. CLEVR_train_000000.png
                image_filename = os.path.join(images_dir, f"CLEVR_train_{int(image_index):06d}.png")
                
                # Iterate over the annotated program steps
                for step in question.get("annotated_program", []):
                    # Only use steps with function "0" and output_values that contain bounding boxes.
                    if step.get("function") == "0" and '[' in step.get("output_values", ""):
                        bboxes = parse_bboxes(step.get("output_values", ""))
                        if not bboxes:
                            continue
                        # Convert each bbox to YOLO format: [x_center, y_center, width, height]
                        yolo_bboxes = [convert_to_yolo(box) for box in bboxes]
                        self.samples.append({
                            "image_path": image_filename,
                            "bboxes": yolo_bboxes
                        })
        logging.info(f"Total YOLO samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Create target tensor of shape (S, S, 5) for YOLO.
        # Each grid cell: [x, y, w, h, conf]
        S = self.S
        target = torch.zeros((S, S, 5))
        for bbox in sample["bboxes"]:
            x_center, y_center, width, height = bbox
            grid_x = int(x_center * S)
            grid_y = int(y_center * S)
            grid_x = min(grid_x, S - 1)
            grid_y = min(grid_y, S - 1)
            # If the cell is already assigned an object, skip additional boxes.
            if target[grid_y, grid_x, 4] == 1:
                continue
            # Coordinates relative to the grid cell.
            cell_x = x_center * S - grid_x
            cell_y = y_center * S - grid_y
            target[grid_y, grid_x, 0] = cell_x
            target[grid_y, grid_x, 1] = cell_y
            target[grid_y, grid_x, 2] = width
            target[grid_y, grid_x, 3] = height
            target[grid_y, grid_x, 4] = 1  # Object confidence
        return image, target

##########################
# YOLO Model Definition
##########################

class YOLO(nn.Module):
    def __init__(self, S=7, B=1, in_channels=3):
        """
        S: Grid size (S x S).
        B: Number of bounding boxes per grid cell (using 1 for simplicity).
        in_channels: Number of input channels (3 for RGB).
        """
        super(YOLO, self).__init__()
        self.S = S
        self.B = B
        # A simple convolutional backbone.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 28 -> 14
        )
        # Additional layer to downsample to a 7x7 grid.
        self.conv_final = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 14 -> 7
        # Fully connected head to output S x S x (B*5) values.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * S * S, 1024),
            nn.ReLU(),
            nn.Linear(1024, S * S * B * 5)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv_final(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5)
        return x

##########################
# YOLO Loss Function
##########################

def yolo_loss(pred, target, lambda_coord=5, lambda_noobj=0.5):
    """
    Computes a simplified YOLO loss.
    Both pred and target have shape (batch, S, S, 5):
      The last dimension represents [x, y, w, h, conf]
    """
    mse = nn.MSELoss(reduction='sum')
    # Masks for grid cells with objects and without objects.
    obj_mask = target[..., 4] > 0
    noobj_mask = target[..., 4] == 0
    
    # Localization loss for cells with objects.
    loss_coord = mse(pred[obj_mask][..., :4], target[obj_mask][..., :4])
    # Confidence loss for cells with objects.
    loss_conf_obj = mse(pred[obj_mask][..., 4], target[obj_mask][..., 4])
    # Confidence loss for cells without objects.
    loss_conf_noobj = mse(pred[noobj_mask][..., 4], target[noobj_mask][..., 4])
    
    loss = lambda_coord * loss_coord + loss_conf_obj + lambda_noobj * loss_conf_noobj
    # Normalize by the batch size.
    batch_size = pred.shape[0]
    return loss / batch_size

##########################
# Training Loop with Early Stopping
##########################

def train(subset_fraction=1.0, patience=10):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Data paths
    annotated_h5 = "annotated_questions.h5"  # HDF5 file with annotations
    images_dir = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/images/train"
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create the dataset with the chosen subset fraction.
    dataset = YOLODataset(annotated_h5, images_dir, transform=transform, subset_fraction=subset_fraction, S=7)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Create the YOLO model.
    model = YOLO(S=7, B=1, in_channels=3)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = yolo_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logging.info(f"No improvement for {epochs_without_improvement} epochs.")
        
        if epochs_without_improvement >= patience:
            logging.info("Early stopping triggered.")
            break
    
    # Save the trained model.
    torch.save(model.state_dict(), "yolo_model.pth")
    logging.info("Training complete. Model saved as yolo_model.pth")

##########################
# Inference Example
##########################

def inference_example():
    """
    Loads the trained YOLO model and runs inference on a sample image.
    The predictions are post-processed to convert cell-relative coordinates to overall normalized coordinates.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    # Create the model and load saved weights.
    model = YOLO(S=7, B=1, in_channels=3)
    model.load_state_dict(torch.load("yolo_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Use the same transformation as training.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # For demonstration, load one sample image.
    images_dir = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/images/train"
    sample_image_path = os.path.join(images_dir, "CLEVR_train_000000.png")
    image = Image.open(sample_image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        prediction = model(image_tensor)  # shape: [1, S, S, 5]
    
    # Postprocess the predictions.
    # Each grid cell outputs [x, y, w, h, conf] where (x, y) is relative to the cell.
    S = 7
    prediction = prediction[0]  # shape: [7, 7, 5]
    threshold = 0.5  # confidence threshold
    boxes = []
    for i in range(S):
        for j in range(S):
            cell_pred = prediction[i, j]
            conf = cell_pred[4].item()
            if conf > threshold:
                # Get cell relative coordinates.
                cell_x = cell_pred[0].item()
                cell_y = cell_pred[1].item()
                width = cell_pred[2].item()
                height = cell_pred[3].item()
                # Convert to overall normalized coordinates.
                x_center = (j + cell_x) / S
                y_center = (i + cell_y) / S
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2
                boxes.append([xmin, ymin, xmax, ymax, conf])
    
    print("Predicted boxes (xmin, ymin, xmax, ymax, confidence):")
    for box in boxes:
        print(box)

##########################
# Main
##########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with a subset of the data and early stopping.")
    parser.add_argument("--subset_fraction", type=float, default=0.01,
                        help="Fraction of the dataset to use for training (default: 0.1)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs with no improvement before stopping early (default: 10)")
    args = parser.parse_args()
    
    train(subset_fraction=args.subset_fraction, patience=args.patience)
    # Run an inference example after training.
    inference_example()
