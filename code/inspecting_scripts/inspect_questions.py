#!/usr/bin/env python3

"""
Simple Decoding Script for CLEVR HDF5 Dataset

This script decodes and displays sample data from the CLEVR HDF5 file using the provided vocabulary JSON.
It converts encoded questions, programs, and answers back into human-readable strings.

Usage:
    python decode_clevr_h5_simple.py --h5_file_path path/to/train_questions.h5 --vocab_json_path path/to/vocab.json --num_samples 5

Arguments:
    --h5_file_path       Path to the HDF5 file (default: 'test_questions.h5')
    --vocab_json_path    Path to the vocabulary JSON file (default: 'data/vocab.json')
    --num_samples        Number of samples to decode and display from both the start and end (default: 5)
"""

import h5py
import json
import numpy as np
import argparse
import os
import sys

# ------------------------------ Utility Functions ------------------------------

def decode_sequence(sequence, idx_to_token, remove_special=True):
    """
    Decode a sequence of indices into a string of tokens.

    Args:
        sequence (list or np.ndarray): Sequence of integer indices.
        idx_to_token (dict): Mapping from index to token.
        remove_special (bool): Whether to remove special tokens like <START> and <END>.

    Returns:
        str: Decoded string.
    """
    # tokens = [idx_to_token.get(idx, '<UNK>') for idx in sequence if idx != 0]  # Ignore <NULL> tokens
    # if remove_special:
    #     tokens = [tok for tok in tokens if tok not in ['<START>', '<END>']]
    # return ' '.join(tokens)
    return sequence

# ------------------------------ Main Function ------------------------------

def load_and_decode_samples(h5_file_path, question_idx_to_token, program_idx_to_token, answer_idx_to_token, num_samples=5):
    """
    Load and decode sample data from the HDF5 file.

    Args:
        h5_file_path (str): Path to the HDF5 file.
        question_idx_to_token (dict): Mapping from question indices to tokens.
        program_idx_to_token (dict): Mapping from program indices to tokens.
        answer_idx_to_token (dict): Mapping from answer indices to tokens.
        num_samples (int): Number of samples to decode and display from both start and end.
    """
    with h5py.File(h5_file_path, 'r') as f:
        # Load total number of samples
        total_samples = f['questions'].shape[0]
        
        # Adjust num_samples if it's more than half of total_samples
        half_total = total_samples // 2
        if num_samples > half_total:
            print(f"Requested num_samples ({num_samples}) is more than half of total samples ({total_samples}). Adjusting num_samples to {half_total}.")
            num_samples = half_total
        
        # Determine indices for first and last num_samples
        first_indices = range(0, num_samples)
        last_indices = range(total_samples - num_samples, total_samples)
        
        # Load datasets
        questions = f['questions']
        programs = f['programs']
        answers = f['answers']
        image_idxs = f['image_idxs']
        orig_idxs = f['orig_idxs']
        question_families = f['question_families'][:] if 'question_families' in f else None

        print("\n--- Decoded Samples: First {} Samples ---\n".format(num_samples))
        for i in first_indices:
            decoded_question = decode_sequence(questions[i], question_idx_to_token)
            decoded_program = decode_sequence(programs[i], program_idx_to_token)
            decoded_answer = answer_idx_to_token.get(answers[i], '<UNK>')

            print(f"Sample {i+1}:")
            print(f"  Original Index: {orig_idxs[i]}")
            print(f"  Image Index: {image_idxs[i]}")
            if question_families is not None:
                print(f"  Question Family: {question_families[i]}")
            print(f"  Question: {decoded_question}")
            print(f"  Program: {decoded_program}")
            print(f"  Answer: {decoded_answer}\n")

        print("\n--- Decoded Samples: Last {} Samples ---\n".format(num_samples))
        for i in last_indices:
            decoded_question = decode_sequence(questions[i], question_idx_to_token)
            decoded_program = decode_sequence(programs[i], program_idx_to_token)
            decoded_answer = answer_idx_to_token.get(answers[i], '<UNK>')

            print(f"Sample {i+1}:")
            print(f"  Original Index: {orig_idxs[i]}")
            print(f"  Image Index: {image_idxs[i]}")
            if question_families is not None:
                print(f"  Question Family: {question_families[i]}")
            print(f"  Question: {decoded_question}")
            print(f"  Program: {decoded_program}")
            print(f"  Answer: {decoded_answer}\n")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Simple Decode CLEVR HDF5 dataset samples.')
    parser.add_argument('--h5_file_path', type=str, default='h5_files/train_questions.h5',
                        help='Path to the HDF5 file (default: test_questions.h5)')
    parser.add_argument('--vocab_json_path', type=str, default='data/vocab.json',
                        help='Path to the vocabulary JSON file (default: data/vocab.json)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to decode and display from both start and end (default: 5)')
    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.h5_file_path):
        print(f"Error: HDF5 file '{args.h5_file_path}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.vocab_json_path):
        print(f"Error: Vocabulary JSON file '{args.vocab_json_path}' does not exist.")
        sys.exit(1)

    # Load the vocabulary JSON
    with open(args.vocab_json_path, 'r') as f:
        vocab = json.load(f)

    # Create inverse mappings: index to token
    question_idx_to_token = {int(idx): token for token, idx in vocab['question_token_to_idx'].items()}
    program_idx_to_token = {int(idx): token for token, idx in vocab['program_token_to_idx'].items()}
    answer_idx_to_token = {int(idx): token for token, idx in vocab['answer_token_to_idx'].items()}

    # Load and decode samples
    load_and_decode_samples(
        h5_file_path=args.h5_file_path,
        question_idx_to_token=question_idx_to_token,
        program_idx_to_token=program_idx_to_token,
        answer_idx_to_token=answer_idx_to_token,
        num_samples=args.num_samples
    )

if __name__ == '__main__':
    main()
