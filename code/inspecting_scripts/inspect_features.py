import h5py

def print_h5_structure(name, obj):
    indent = '  ' * (name.count('/') - 1)
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")

# Path to your .h5 file
# h5_file_path = '/Users/guoyuzhang/University/Y5/diss/code/h5_files/train_questions.h5'
h5_file_path = '/Users/guoyuzhang/University/Y5/diss/clevr-iep/data/train_features.h5'

# Open the HDF5 file in read mode
with h5py.File(h5_file_path, 'r') as h5file:
    print("HDF5 File Structure:")
    h5file.visititems(print_h5_structure)
