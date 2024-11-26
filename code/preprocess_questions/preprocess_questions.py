#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse

import json
import os

import h5py
import numpy as np

import utils_programs
from utils_preprocess import tokenize, encode


"""
Preprocessing script for CLEVR question files.
"""


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='postfix')
parser.add_argument('--input_questions_json', required=True)
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)
parser.add_argument('--output_h5_file', required=True)


def program_to_str(program, mode):
  if mode == 'chain':
    if not utils_programs.is_chain(program):
      return None
    return utils_programs.list_to_str(program)
  elif mode == 'prefix':
    program_prefix = utils_programs.list_to_prefix(program)
    return utils_programs.list_to_str(program_prefix)
  elif mode == 'postfix':
    program_postfix = utils_programs.list_to_postfix(program)
    return utils_programs.list_to_str(program_postfix)
  return None

def main(args):
  if args.input_vocab_json == '':
    print('Must give --input_vocab_json')
    return

  print('Loading data')
  with open(args.input_questions_json, 'r') as f:
    questions = json.load(f)['questions']

  print('Loading vocab')
  with open(args.input_vocab_json, 'r') as f:
    vocab = json.load(f)

  # Encode all questions and programs
  print('Encoding data')
  questions_encoded = []
  programs_encoded = []
  question_families = []
  orig_idxs = []
  image_idxs = []
  answers = []
  for orig_idx, q in enumerate(questions):
    question = q['question']

    orig_idxs.append(orig_idx)
    image_idxs.append(q['image_index'])
    if 'question_family_index' in q:
      question_families.append(q['question_family_index'])
    
    # print("question", question)
    question_tokens = tokenize(question,
                        punct_to_keep=[';', ','],
                        punct_to_remove=['?', '.'])
    # print("question_tokens", question_tokens)
    question_encoded = encode(question_tokens,
                         vocab['question_token_to_idx'])
    # print("question_encoded", question_encoded)
    questions_encoded.append(question_encoded)

    if 'program' in q:
      program = q['program']
      # print(program)
      program_str = program_to_str(program, args.mode)
      program_tokens = tokenize(program_str)
      program_encoded = encode(program_tokens, vocab['program_token_to_idx'])
      programs_encoded.append(program_encoded)

    if 'answer' in q:
      answers.append(vocab['answer_token_to_idx'][q['answer']])

  # Pad encoded questions and programs
  max_question_length = max(len(x) for x in questions_encoded)
  for qe in questions_encoded:
    while len(qe) < max_question_length:
      qe.append(vocab['question_token_to_idx']['<NULL>'])

  if len(programs_encoded) > 0:
    max_program_length = max(len(x) for x in programs_encoded)
    for pe in programs_encoded:
      while len(pe) < max_program_length:
        pe.append(vocab['program_token_to_idx']['<NULL>'])

  # Create h5 file
  print('Writing output')
  questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
  programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
  print(questions_encoded.shape)
  print(programs_encoded.shape)
  with h5py.File(args.output_h5_file, 'w') as f:
    f.create_dataset('questions', data=questions_encoded)
    f.create_dataset('image_idxs', data=np.asarray(image_idxs))
    f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))

    if len(programs_encoded) > 0:
      f.create_dataset('programs', data=programs_encoded)
    if len(question_families) > 0:
      f.create_dataset('question_families', data=np.asarray(question_families))
    if len(answers) > 0:
      f.create_dataset('answers', data=np.asarray(answers))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
