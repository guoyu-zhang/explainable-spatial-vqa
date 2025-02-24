import json
import h5py
import os

def load_vocab(vocab_path):
    """
    Loads the vocabulary mapping (token -> index) from a JSON file.
    """
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab

def decode_if_bytes(value):
    """
    Decodes a bytes object to a UTF-8 string if necessary.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value

def load_h5_data(h5_path, vocab):
    """
    Loads the HDF5 file containing the annotated questions and converts text fields
    to token indices using the provided vocabulary.
    
    Returns:
        A list of dictionaries (one per question) with keys:
          - image_index, question_index, question, answer,
          - question_tokens, answer_tokens,
          - annotated_program (each step with keys: inputs, input_values, output_values, chain_of_thought).
    """
    data = []
    with h5py.File(h5_path, "r") as h5f:
        # Access the "questions" group.
        questions_group = h5f["questions"]
        
        # Iterate over each question group (e.g., "q_0000", "q_0001", ...).
        for q_key in sorted(questions_group.keys()):
            q_group = questions_group[q_key]
            question_entry = {}
            
            # Get the basic fields.
            question_entry["image_index"] = int(q_group["image_index"][()])
            question_entry["question_index"] = int(q_group["question_index"][()])
            
            question_str = decode_if_bytes(q_group["question"][()])
            answer_str = decode_if_bytes(q_group["answer"][()])
            question_entry["question"] = question_str
            question_entry["answer"] = answer_str
            
            # Tokenize the question and answer, mapping tokens to indices.
            question_entry["question_tokens"] = [
                vocab.get(tok, vocab.get("<UNK>")) for tok in question_str.strip().split()
            ]
            question_entry["answer_tokens"] = [
                vocab.get(tok, vocab.get("<UNK>")) for tok in answer_str.strip().split()
            ]
            
            # Process the annotated program.
            annotated_program = []
            ap_group = q_group["annotated_program"]
            for step_key in sorted(ap_group.keys()):
                step_group = ap_group[step_key]
                step = {}
                
                # Each of these fields was stored as a JSON string.
                inputs_str = decode_if_bytes(step_group["inputs"][()])
                input_values_str = decode_if_bytes(step_group["input_values"][()])
                output_values_str = decode_if_bytes(step_group["output_values"][()])
                chain_str = decode_if_bytes(step_group["chain_of_thought"][()])
                
                # Convert JSON strings back to Python objects.
                step["inputs"] = json.loads(inputs_str)
                step["input_values"] = json.loads(input_values_str)
                step["output_values"] = json.loads(output_values_str)
                chain_list = json.loads(chain_str)
                
                # Convert each token in the chain of thought to its vocabulary index.
                step["chain_of_thought"] = [
                    vocab.get(token, vocab.get("<UNK>")) for token in chain_list
                ]
                
                annotated_program.append(step)
            
            question_entry["annotated_program"] = annotated_program
            data.append(question_entry)
    return data

def main():
    # Paths to the HDF5 file and the vocabulary file.
    h5_path = "output_full_annotations.h5"
    vocab_path = "full_annotations_vocab.json"
    
    # Load the vocabulary.
    vocab = load_vocab(vocab_path)
    print(f"Loaded vocabulary with {len(vocab)} tokens.\n")
    
    # Load the HDF5 data (assumes there are 5 questions).
    questions = load_h5_data(h5_path, vocab)
    print(f"Loaded {len(questions)} questions from the HDF5 file.\n")
    
    # Print out the loaded information for each question.
    for q in questions:
        print("=======================================")
        print(f"Question Index: {q['question_index']}")
        print(f"Image Index: {q['image_index']}")
        print(f"Question: {q['question']}")
        print(f"Question Tokens: {q['question_tokens']}")
        print(f"Answer: {q['answer']}")
        print(f"Answer Tokens: {q['answer_tokens']}")
        print("Annotated Program:")
        for step in q["annotated_program"]:
            print("  Inputs:", step["inputs"])
            print("  Input Values:", step["input_values"])
            print("  Output Values:", step["output_values"])
            print("  Chain of Thought (token indices):", step["chain_of_thought"])
        print("=======================================\n")

if __name__ == "__main__":
    main()
