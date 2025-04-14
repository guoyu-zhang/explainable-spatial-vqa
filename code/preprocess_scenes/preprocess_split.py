import json
import h5py
import os

def save_questions_separately(questions, output_h5_path):
    """
    Save each question as a separate group in the HDF5 file.
    Each question group will have datasets for each key (e.g., image_index, question, answer, etc.),
    with the value stored as a JSON string.
    """
    with h5py.File(output_h5_path, 'w') as hf:
        q_group = hf.create_group("questions")
        for i, question in enumerate(questions):
            grp = q_group.create_group(f"question_{i}")
            for key, value in question.items():
                # Convert the value to JSON so that even nested structures are stored as strings.
                json_value = json.dumps(value)
                # Create a dataset for each key using a string data type.
                dt = h5py.string_dtype(encoding='utf-8')
                grp.create_dataset(key, data=json_value, dtype=dt)
    print(f"Saved {len(questions)} questions in separate groups to {output_h5_path}")

def main():
    # Path to the JSON file with annotated questions.
    json_input_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions1.json"
    output_h5_path = "annotated_questions_separated.h5"
    
    # Load the annotated questions.
    with open(json_input_path, 'r') as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    print(f"Total questions to save: {len(questions)}")
    
    # Save the questions with separate keys.
    save_questions_separately(questions, output_h5_path)

if __name__ == "__main__":
    main()
