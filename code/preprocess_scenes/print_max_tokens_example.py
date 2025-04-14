import json
import re
import logging
import h5py

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"

def find_examples_with_n_tokens(n=10):
    with h5py.File(ANNOTATED_QUESTIONS_H5, 'r') as hf:
        data = hf['questions'][()]
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        annotated_questions = json.loads(data)
    
    examples = []
    # Iterate over all questions and their program steps.
    for q in annotated_questions:
        for step in q.get("annotated_program", []):
            input_val = step.get("input_values", "").strip()
            # Only consider examples that are not bounding boxes.
            if '[' in input_val or input_val == "":
                continue
            tokens = input_val.split()
            if len(tokens) == n:
                examples.append({
                    "question": q.get("question", ""),
                    "question_index": q.get("question_index"),
                    "input_values": input_val,
                    "annotated_program_step": step
                })
    return examples

def main():
    n = 10
    examples = find_examples_with_n_tokens(n)
    if examples:
        logging.info(f"Found {len(examples)} examples with {n} tokens in input_values. Here's the first one:")
        print(json.dumps(examples[0], indent=4))
    else:
        logging.info(f"No examples found with {n} tokens in input_values.")

if __name__ == "__main__":
    main()
