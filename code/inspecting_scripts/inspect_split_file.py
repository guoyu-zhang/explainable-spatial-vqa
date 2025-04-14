import h5py

def print_h5_structure(file_path):
    with h5py.File(file_path, 'r') as hf:
        print("File Structure:")
        for key in hf.keys():
            print(f"- {key} (Group)")
            group = hf[key]
            for subkey in group.keys():
                item = group[subkey]
                if isinstance(item, h5py.Group):
                    print(f"  - {subkey} (Group)")
                else:
                    print(f"  - {subkey} (Dataset)")
        
        # Print structure for only the first 5 question groups in the "questions" group.
        if 'questions' in hf:
            print("\nStructure of the first 5 question entries in 'questions':")
            q_group = hf['questions']
            question_keys = sorted(q_group.keys())
            for question_key in question_keys[:5]:
                print(f"Question Group: {question_key}")
                question_group = q_group[question_key]
                for dataset_key in question_group.keys():
                    print(f"  - {dataset_key} (Dataset)")
        else:
            print("\nNo 'questions' group found in the file.")

def main():
    file_path = "annotated_questions_separated.h5"  # Update the path if needed.
    print("Annotated Questions Separated HDF5 File Structure:")
    print_h5_structure(file_path)

if __name__ == "__main__":
    main()
