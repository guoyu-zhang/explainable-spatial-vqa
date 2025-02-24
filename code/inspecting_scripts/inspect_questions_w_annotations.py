#!/usr/bin/env python3

import json
import h5py
import numpy as np

def load_id_to_token(vocab_json_path):
    """
    Loads the vocabulary JSON file, which has:
      {
        "token_to_id": { ... },
        "id_to_token": { "0": "<PAD>", "1": "<UNK>", ... }
      }
    Returns a dictionary id_to_token with *integer* keys 
    and string token values.
    """
    with open(vocab_json_path, 'r') as f:
        vocab_data = json.load(f)
    id_to_token_str_keys = vocab_data['id_to_token']  # e.g. {"0":"<PAD>", "1":"<UNK>", ...}
    # Convert str keys to int
    id_to_token = {}
    for kstr, token_str in id_to_token_str_keys.items():
        id_to_token[int(kstr)] = token_str
    return id_to_token

def decode_sequence(seq_ids, id_to_token):
    """
    Given a 1D array/list of integer token IDs, returns a list of string tokens
    using id_to_token. If something is not in id_to_token, we can fallback to <UNK>.
    """
    result_tokens = []
    for token_id in seq_ids:
        if token_id in id_to_token:
            result_tokens.append(id_to_token[token_id])
        else:
            result_tokens.append("<UNK>")
    return result_tokens

def inspect_h5_with_vocab(h5_path, vocab_json_path):
    """
    Loads the HDF5 file, the vocab, and prints first 5 and last 5 entries,
    decoding question_tokens, answer_tokens, program_tokens back into strings.
    """
    # 1) Load the vocab
    id_to_token = load_id_to_token(vocab_json_path)

    with h5py.File(h5_path, 'r') as f:
        # 2) Read datasets
        image_index = f['image_index'][:]             # shape (N,)
        question_tokens = f['question_tokens'][:]     # shape (N, QLEN)
        answer_tokens = f['answer_tokens'][:]         # shape (N, ALEN)
        program_tokens = f['program_tokens'][:]       # shape (N, PLEN)

        N = image_index.shape[0]
        print(f"[inspect_h5_with_vocab] Found {N} entries in {h5_path}.")

        num_front = min(5, N)  # first 5
        num_back = min(5, N)   # last 5

        # 3) Print first 5
        print("\n----- First 5 entries -----\n")
        for i in range(num_front):
            print(f"Index {i} (image_index={image_index[i]}):")
            # decode question, answer, program
            q_decoded = decode_sequence(question_tokens[i], id_to_token)
            a_decoded = decode_sequence(answer_tokens[i], id_to_token)
            p_decoded = decode_sequence(program_tokens[i], id_to_token)

            print("  question_tokens:", q_decoded)
            print("  answer_tokens:", a_decoded)
            print("  program_tokens:", p_decoded)
            print()

        # 4) Print last 5
        print("\n----- Last 5 entries -----\n")
        for i in range(N - num_back, N):
            print(f"Index {i} (image_index={image_index[i]}):")
            q_decoded = decode_sequence(question_tokens[i], id_to_token)
            a_decoded = decode_sequence(answer_tokens[i], id_to_token)
            p_decoded = decode_sequence(program_tokens[i], id_to_token)

            print("  question_tokens:", q_decoded)
            print("  answer_tokens:", a_decoded)
            print("  program_tokens:", p_decoded)
            print()

def main():
    h5_file = "mapped_sequences.h5"     # your HDF5 file
    vocab_json = "vocab.json"           # your vocab JSON
    inspect_h5_with_vocab(h5_file, vocab_json)

if __name__ == "__main__":
    main()
