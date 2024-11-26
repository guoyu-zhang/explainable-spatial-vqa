import argparse
import os
import h5py
import numpy as np
from PIL import Image  # For image resizing and conversion
import torch
import torchvision


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)


# Function to build the model
def build_model(args):
    if not hasattr(torchvision.models, args.model):
        raise ValueError(f'Invalid model "{args.model}"')
    if 'resnet' not in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    # print(cnn)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(args.model_stage):
        name = f'layer{i + 1}'
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    # print(model)
    # Set device: MPS (Metal Performance Shaders) if available, otherwise CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    return model, device


# Function to process a batch of images
def run_batch(cur_batch, model, device):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).to(device)

    with torch.no_grad():
        feats = model(image_batch)
    feats = feats.cpu().numpy()

    return feats


def main(args):
    input_paths = []
    idx_set = set()

    # Collect input image paths
    for fn in os.listdir(args.input_image_dir):
        if not fn.endswith('.png'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        input_paths.append((os.path.join(args.input_image_dir, fn), idx))
        idx_set.add(idx)

    input_paths.sort(key=lambda x: x[1])
    if len(input_paths) == 0:
        raise ValueError("No valid images found in the input directory.")

    assert len(idx_set) == len(input_paths)
    assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1

    if args.max_images is not None:
        input_paths = input_paths[:args.max_images]
    
    print(f"Processing images from {input_paths[0][0]} to {input_paths[-1][0]}")

    model, device = build_model(args)
    img_size = (args.image_height, args.image_width)

    with h5py.File(args.output_h5_file, 'w') as f:
        feat_dset = None  # Initialize dataset as None
        i0 = 0
        cur_batch = []

        # Process images
        for i, (path, idx) in enumerate(input_paths):
            print(f"Processing image {path}")

            # Open the image using Pillow
            img = Image.open(path)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize the image
            img = img.resize(img_size, Image.BICUBIC)

            # Convert to NumPy array and adjust dimensions for batching
            img = np.array(img).transpose(2, 0, 1)[None]  # Adds a new axis at the beginning to represent the batch dimension, resulting in a shape of (1, channels, height, width). Changes the image array shape from (height, width, channels) to (channels, height, width), which is the expected input format for PyTorch models.
            cur_batch.append(img)

            # Process batch when it's full
            if len(cur_batch) == args.batch_size:
                feats = run_batch(cur_batch, model, device)
                if feat_dset is None:
                    # Initialize the feature dataset
                    N = len(input_paths)
                    _, C, H, W = feats.shape
                    feat_dset = f.create_dataset('features', (N, C, H, W), dtype=np.float32)
                i1 = i0 + len(cur_batch)
                feat_dset[i0:i1] = feats
                i0 = i1
                print(f'Processed {i1} / {len(input_paths)} images')
                cur_batch = []

        # Process remaining images in the final batch
        if len(cur_batch) > 0:
            feats = run_batch(cur_batch, model, device)
            if feat_dset is None:
                # Initialize the feature dataset (edge case: very small dataset)
                N = len(input_paths)
                _, C, H, W = feats.shape
                feat_dset = f.create_dataset('features', (N, C, H, W), dtype=np.float32)
            i1 = i0 + len(cur_batch)
            feat_dset[i0:i1] = feats
            print(f'Processed {i1} / {len(input_paths)} images')

        print(f"Features saved to {args.output_h5_file}")



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
