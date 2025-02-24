#!/usr/bin/env python3

import h5py
import numpy as np

# Change this path to your .h5 file (from the bounding-box script)
H5_FILE_PATH = "h5_files/train_scenes.h5"  # Change to your .h5 file path

def print_dataset_preview(ds_name, ds_data):
    """
    Print shape, dtype, and first/last 5 entries along axis=0.

    ds_data is a NumPy array loaded from the dataset.
    """
    print(f"\n=== Dataset: {ds_name} ===")
    print(f"  Shape: {ds_data.shape}, Dtype: {ds_data.dtype}")

    if ds_data.ndim == 1:
        # This could be 'image_index' or 'image_filename' (strings).
        print("  First 5 entries:", ds_data[:5])
        print("  Last 5 entries: ", ds_data[-5:])
    elif ds_data.ndim >= 2:
        # For bounding_boxes (3D or 2D?), class_labels (2D).
        # Show the first 5 rows along axis=0, and the last 5 rows along axis=0.
        print("  First 5 rows:\n", ds_data[:5], "\n")
        print("  Last 5 rows:\n", ds_data[-5:], "\n")

def main():
    with h5py.File(H5_FILE_PATH, "r") as f:
        # 1) Print any file-level attributes
        print("=== File-Level Attributes ===")
        if len(f.attrs) == 0:
            print("  (No file-level attributes found.)")
        else:
            for attr_name, attr_val in f.attrs.items():
                print(f"  {attr_name}: {attr_val}")

        # 2) Print dataset information and samples
        print("\n=== Datasets in File ===")
        for ds_name in f.keys():
            ds = f[ds_name]
            # Load the data into memory
            ds_data = ds[()]  # same as ds[:]
            print_dataset_preview(ds_name, ds_data)

if __name__ == "__main__":
    main()
