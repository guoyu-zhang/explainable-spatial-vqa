import json
import h5py

def reverse_tokens(tokenized_str, field, inv_vocab):
    """
    Given a space-separated string of token indices,
    look up each index in the inverse vocabulary mapping to recover the original token.
    """
    if not tokenized_str.strip():
        return ""
    tokens = tokenized_str.split()
    original_tokens = [inv_vocab.get(token, token) for token in tokens]
    return " ".join(original_tokens)

def reverse_question_conversion(question, inv_vocab):
    """
    Convert back all fields in the question from indices to tokens.
    Expected fields: 'answer', 'final_chain_of_thought', and an 'annotated_program'
    (with 'function', 'input_values', and 'output_values').
    """
    if "answer" in question:
        question["answer"] = reverse_tokens(question["answer"], "other", inv_vocab)
    if "final_chain_of_thought" in question:
        question["final_chain_of_thought"] = [
            reverse_tokens(chain, "other", inv_vocab)
            for chain in question["final_chain_of_thought"]
        ]
    if "annotated_program" in question:
        for step in question["annotated_program"]:
            if "function" in step:
                # The "function" field was not split on '[' or ']'
                step["function"] = reverse_tokens(step["function"], "function", inv_vocab)
            if "input_values" in step:
                step["input_values"] = reverse_tokens(step["input_values"], "other", inv_vocab)
            if "output_values" in step:
                step["output_values"] = reverse_tokens(step["output_values"], "other", inv_vocab)
    return question

def print_h5_description(h5_file_path):
    """
    Prints out the top-level keys in the HDF5 file and some basic info about each dataset.
    """
    with h5py.File(h5_file_path, 'r') as hf:
        print("Datasets in the file:", list(hf.keys()))
        for key in hf.keys():
            data = hf[key][()]
            print(f"\nDataset: {key}")
            print(f"Type: {type(data)}")
            if isinstance(data, bytes):
                print("This is a byte string (likely JSON-encoded).")
            else:
                print("Data:", data)

def main():
    h5_file = "/Users/guoyuzhang/University/Y5/diss/vqa/code/annotated_questions_with_vocab.h5"
    
    # Print HDF5 file description
    print("HDF5 File Description:")
    print_h5_description(h5_file)
    
    # Open the HDF5 file and read the JSON strings for questions and vocabulary.
    with h5py.File(h5_file, 'r') as hf:
        questions_json = hf["questions"][()]
        vocab_json = hf["vocab"][()]
    
    if isinstance(questions_json, bytes):
        questions_json = questions_json.decode('utf-8')
    if isinstance(vocab_json, bytes):
        vocab_json = vocab_json.decode('utf-8')
    
    questions_obj = json.loads(questions_json)
    vocab = json.loads(vocab_json)
    
    # Build the inverse vocabulary: mapping from index (as string) back to token.
    inv_vocab = {str(index): token for token, index in vocab.items()}
    
    questions = questions_obj["questions"]
    total = len(questions)
    print(f"\nTotal number of questions in file: {total}")
    
    # Part 1: Reverse conversion for demonstration (first two and last two questions)
    selected_questions_decoded = questions[:2] + questions[-2:]
    reversed_questions = [reverse_question_conversion(q, inv_vocab) for q in selected_questions_decoded]
    print("\nReversed Questions (first two and last two):")
    print(json.dumps(reversed_questions, indent=2))
    
    # Part 2: Output raw questions (first five and last five) without decoding
    selected_questions_raw = questions[:2] + questions[-2:]
    print("\nRaw Questions (first five and last five without decoding):")
    print(json.dumps(selected_questions_raw, indent=2))

if __name__ == "__main__":
    main()
