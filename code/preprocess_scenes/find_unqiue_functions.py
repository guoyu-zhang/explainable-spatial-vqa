import json
import h5py

ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"

def get_unique_functions(h5_file_path):
    # Open the HDF5 file and load the JSON list of question objects.
    with h5py.File(h5_file_path, 'r') as hf:
        data = hf['questions'][()]
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        annotated_questions = json.loads(data)
    
    unique_funcs = set()
    # Iterate over each question and each step in the annotated program.
    for question in annotated_questions:
        for step in question.get("annotated_program", []):
            func = step.get("function", "").strip()
            if func:
                unique_funcs.add(func)
    return unique_funcs

def main():
    unique_funcs = get_unique_functions(ANNOTATED_QUESTIONS_H5)
    print(f"Number of unique function tokens: {len(unique_funcs)}")
    print("Unique tokens:", unique_funcs)

if __name__ == "__main__":
    main()
