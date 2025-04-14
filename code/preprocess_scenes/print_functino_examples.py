import json
import logging
import h5py

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

VOCAB_PATH = "/Users/guoyuzhang/University/Y5/diss/vqa/code/vocab2.json"
ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab

def load_annotated_questions(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        data = hf['questions'][()]
        # Decode bytes to string if necessary
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        questions = json.loads(data)
    return questions

def group_steps_by_function(questions):
    # Create a dictionary to group examples by their function token.
    groups = {}
    for q in questions:
        q_text = q.get("question", "")
        for step in q.get("annotated_program", []):
            func_token = step.get("function", "").strip()
            if not func_token:
                continue
            groups.setdefault(func_token, []).append({
                "question": q_text,
                "step": step
            })
    return groups

def print_function_examples(groups, vocab, max_examples=3):
    for func_token, examples in groups.items():
        vocab_index = vocab.get(func_token, "Not Found")
        logging.info(f"Function token: '{func_token}' | Vocab index: {vocab_index} | Count: {len(examples)}")
        for i, ex in enumerate(examples[:max_examples]):
            logging.info(f"  Example {i+1}:")
            logging.info(f"    Question: {ex['question']}")
            logging.info(f"    Step details: {json.dumps(ex['step'], indent=2)}")
        logging.info("-" * 50)

def main():
    vocab = load_vocab(VOCAB_PATH)
    questions = load_annotated_questions(ANNOTATED_QUESTIONS_H5)
    groups = group_steps_by_function(questions)
    print_function_examples(groups, vocab, max_examples=3)

if __name__ == "__main__":
    main()
