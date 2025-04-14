import json
import h5py
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def read_first_few_objects(h5_file_path: str, n: int = 5):
    """
    Opens the HDF5 file at `h5_file_path`, reads the 'questions' dataset,
    and returns the first n question objects as a list.
    """
    with h5py.File(h5_file_path, 'r') as hf:
        # Read the dataset containing the questions JSON string.
        questions_data = hf['questions'][()]
        # If the data is in bytes, decode it.
        if isinstance(questions_data, bytes):
            questions_data = questions_data.decode('utf-8')
        # Parse the JSON string.
        questions = json.loads(questions_data)
    return questions[:n]

def main():
    h5_file_path = "annotated_questions.h5"  # adjust path if necessary
    logging.info(f"Reading the first few question objects from {h5_file_path}...")
    first_objects = read_first_few_objects(h5_file_path, n=5)
    # Pretty-print the first few question objects
    print(json.dumps(first_objects, indent=4))

if __name__ == "__main__":
    main()
