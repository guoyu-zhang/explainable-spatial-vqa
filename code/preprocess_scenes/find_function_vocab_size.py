#!/usr/bin/env python3
import json
import re
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def canonicalize(token: str) -> str:
    token_lower = token.lower()
    if token_lower in {"yes", "true"}:
        return "true"
    if token_lower in {"no", "false"}:
        return "false"
    return token_lower

def tokenize_function(text: str) -> list:
    # For function fields, we assume the entire text is a single token.
    # You can adjust this if you need a more elaborate tokenization.
    return [text] if text else []

def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_function_vocab.py <annotated_questions.json>")
        sys.exit(1)
    json_path = sys.argv[1]
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    vocab = {}
    next_index = 0
    # Iterate over each question and each step in its annotated program.
    for question in data:
        annotated_program = question.get("annotated_program", [])
        for step in annotated_program:
            func_text = step.get("function", "").strip()
            tokens = tokenize_function(func_text)
            for token in tokens:
                can_tok = canonicalize(token)
                if can_tok not in vocab:
                    vocab[can_tok] = next_index
                    next_index += 1
    
    print("Function vocabulary size:", len(vocab))
    print("Vocabulary mapping:")
    for token, index in vocab.items():
        print(f"  {token}: {index}")

if __name__ == "__main__":
    main()
