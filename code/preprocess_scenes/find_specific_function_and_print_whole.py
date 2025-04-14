import json
import logging
import h5py

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"

def load_questions(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        data = hf['questions'][()]
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        questions = json.loads(data)
    return questions

def print_question_for_function(function_value="40"):
    questions = load_questions(ANNOTATED_QUESTIONS_H5)
    for q in questions:
        for step in q.get("annotated_program", []):
            if step.get("function") == function_value:
                # Print the entire question object
                print(json.dumps(q, indent=4))
                return
    print(f"No question object found containing function {function_value}")

def main():
    print_question_for_function("40")

if __name__ == "__main__":
    main()
