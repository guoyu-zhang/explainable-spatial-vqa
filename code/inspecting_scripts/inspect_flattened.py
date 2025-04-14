import h5py

def print_first_five_question_groups(file_path):
    """
    Opens the flattened HDF5 file and prints out the contents (datasets) of the first five
    top-level question groups.
    """
    with h5py.File(file_path, 'r') as hf:
        # Get the first five keys (question groups) sorted alphabetically.
        question_keys = sorted(hf.keys())[:5]
        print("First 5 question groups:")
        for key in question_keys:
            print(f"\nQuestion Group: {key}")
            group = hf[key]
            for dataset_key in group.keys():
                try:
                    value = group[dataset_key][()]
                    # Decode bytes if necessary
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    print(f"  {dataset_key}: {value}")
                except Exception as e:
                    print(f"  {dataset_key}: Could not read data (error: {e})")

def main():
    file_path = "flattened_questions.h5"  # Update the file path if necessary.
    print_first_five_question_groups(file_path)

if __name__ == "__main__":
    main()
