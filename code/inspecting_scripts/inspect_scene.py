import h5py
import numpy as np

H5_PATH = "h5_files/train_scenes.h5"  # Change to your .h5 file path

def print_dataset_preview(ds_name, ds_data):
    """
    Prints the first 5 and last 5 “rows” of ds_data along dim 0.
    ds_data is a NumPy array loaded from the dataset.
    """
    print(f"Dataset name: {ds_name}")
    print(f"  Shape: {ds_data.shape}, Dtype: {ds_data.dtype}")
    
    # If the dataset is 1D or 2D or 3D, etc.
    if ds_data.ndim == 1:
        print("  First 5 entries:", ds_data[:5])
        print("  Last 5 entries: ", ds_data[-5:])
    elif ds_data.ndim >= 2:
        print("  First 5 rows:\n", ds_data[:5], "\n")
        print("  Last 5 rows:\n", ds_data[-5:], "\n")
    # If it has more dimensions, you can adapt accordingly.

def main():
    with h5py.File(H5_PATH, "r") as f:
        # ---------------------------------------------------------------------
        # 1. Print file-level attributes (if any)
        # ---------------------------------------------------------------------
        print("=== File-level Attributes ===")
        for attr_name, attr_val in f.attrs.items():
            print(f"{attr_name}: {attr_val}")

        # ---------------------------------------------------------------------
        # 2. Inspect each dataset
        # ---------------------------------------------------------------------
        print("\n=== Datasets in File ===")
        for ds_name in f.keys():
            ds = f[ds_name]
            # Read the data into memory (caution if your dataset is very large!)
            ds_data = ds[()]  # same as ds[:]
            
            # Print basic info + first/last 5 rows
            print_dataset_preview(ds_name, ds_data)

if __name__ == "__main__":
    main()
