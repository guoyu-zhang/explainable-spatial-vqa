import os
import argparse
import logging
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
# Prediction & Drawing Functions
##########################

def predict_boxes(image_tensor, model, threshold=0.5):
    """
    Runs the model on the image tensor (shape: [1, 3, 224, 224]) and
    returns a list of predicted boxes. Each box is in normalized coordinates:
    [xmin, ymin, xmax, ymax, confidence].
    """
    with torch.no_grad():
        prediction = model(image_tensor)  # shape: [1, S, S, 5]
    S = model.S
    prediction = prediction[0]  # shape: [S, S, 5]
    boxes = []
    for i in range(S):
        for j in range(S):
            cell_pred = prediction[i, j]
            conf = cell_pred[4].item()
            if conf > threshold:
                cell_x = cell_pred[0].item()
                cell_y = cell_pred[1].item()
                width = cell_pred[2].item()
                height = cell_pred[3].item()
                # Convert cell-relative coordinates to overall normalized coordinates.
                x_center = (j + cell_x) / S
                y_center = (i + cell_y) / S
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2
                boxes.append([xmin, ymin, xmax, ymax, conf])
    return boxes

def draw_boxes(image, boxes):
    """
    Draws bounding boxes on a PIL image.
    The image is assumed to be 224x224 (same as the model input size).
    Boxes are given in normalized coordinates.
    """
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for box in boxes:
        xmin, ymin, xmax, ymax, conf = box
        left = int(xmin * w)
        top = int(ymin * h)
        right = int(xmax * w)
        bottom = int(ymax * h)
        # Skip invalid boxes
        if right < left or bottom < top:
            logging.warning(f"Skipping invalid box: left={left}, top={top}, right={right}, bottom={bottom}")
            continue
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
    return image

##########################
# Main Prediction Script
##########################

def main(args):
    device = torch.device("mps")
    logging.info(f"Using device: {device}")
    
    # Load the trained model.
    model = YOLO(S=7, B=1, in_channels=3)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define the transformation (same as used during training).
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # List images in the input folder.
    image_files = [f for f in os.listdir(args.input_folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files = image_files[:args.num_images]
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    for img_file in image_files:
        img_path = os.path.join(args.input_folder, img_file)
        image = Image.open(img_path).convert("RGB")
        # Resize image for model input.
        image_resized = transform(image)
        image_tensor = image_resized.unsqueeze(0).to(device)
        # Predict boxes.
        boxes = predict_boxes(image_tensor, model, threshold=args.threshold)
        # Convert tensor back to PIL image.
        image_with_boxes = transforms.ToPILImage()(image_resized.cpu())
        image_with_boxes = draw_boxes(image_with_boxes, boxes)
        output_path = os.path.join(args.output_folder, img_file)
        image_with_boxes.save(output_path)
        logging.info(f"Saved predicted image to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict bounding boxes on images and save them.")
    parser.add_argument("--model_path", type=str, default="yolo_model.pth",
                        help="Path to the trained YOLO model weights.")
    parser.add_argument("--input_folder", type=str, default="/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/images/train",
                        help="Folder containing the input images.")
    parser.add_argument("--output_folder", type=str, default="predicted_images",
                        help="Folder to save the output images with drawn bounding boxes.")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images to process.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for drawing boxes.")
    args = parser.parse_args()
    
    main(args)
