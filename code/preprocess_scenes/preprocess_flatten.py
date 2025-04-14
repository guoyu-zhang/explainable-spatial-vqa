import h5py

def flatten_questions(input_file, output_file):
    """
    Opens the input HDF5 file, reads the 'questions' group, and writes its children (question groups)
    to the output HDF5 file at the root level.
    """
    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dst:
        if 'questions' not in src:
            print("No 'questions' group found in the source file.")
            return
        questions_group = src['questions']
        for key in questions_group.keys():
            # Copy each question group to the root level of the destination file.
            dst.copy(questions_group[key], key)
        print("Flattening complete. Output written to:", output_file)

def main():
    input_file = "annotated_questions_separated.h5"  # Update path if necessary.
    output_file = "flattened_questions.h5"            # New file with flattened structure.
    flatten_questions(input_file, output_file)

if __name__ == "__main__":
    main()
