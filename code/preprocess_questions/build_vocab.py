import json
import logging
import argparse
import re

def setup_logging():
    """
    Sets up the logging configuration.
    Logs are printed to the console with the INFO level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_json(file_path):
    """
    Loads a JSON file and returns its content as a Python variable.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data as a Python dictionary or list.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            logging.info(f"Successfully loaded JSON data from '{file_path}'.")
            return data
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        logging.error(f"Error: Failed to parse JSON file. {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

def parse_arguments():
    """
    Parses command-line arguments for input and output file paths.

    :return: Parsed arguments with 'input' and 'output' attributes.
    """
    parser = argparse.ArgumentParser(description='Build a vocabulary dictionary from CLEVR JSON questions.')
    parser.add_argument('--input', type=str, help='Path to the input JSON file.')
    parser.add_argument('--output', type=str, default='vocab.json', help='Path to the output vocab JSON file.')
    return parser.parse_args()

def tokenize(text, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None):
    """
    Tokenizes the input text into words (including contractions and possessives) and punctuation,
    treating punctuation as separate tokens.
    
    :param text: The text string to tokenize.
    :return: A list of word and punctuation tokens.
    """
    # Regex pattern to match words with optional apostrophes and separate punctuation
    pattern = r"\w+(?:'\w+)?|[^\w\s.?]"
    tokens = re.findall(pattern, text)
    return tokens


def main():
    """
    Main function to load JSON data, build the vocab dictionary, and write it to a JSON file.
    Includes combined 'function' and 'value_inputs' as vocabulary entries.
    """
    setup_logging()
    args = parse_arguments()
    program_vocab = {"<NULL>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    program_counter = 4
    answer_vocab = {"<NULL>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    answer_counter = 4
    question_vocab = {"<NULL>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    question_counter = 4
    
    # json_file_path = args.input
    # json_file_paths = ['CLEVR_val_questions_first.json', 
    #                    'CLEVR_test_questions_first.json', 
    #                    'CLEVR_train_questions_first.json']
    json_file_paths = ['data/CLEVR_v1.0/questions/CLEVR_val_questions.json', 
                       'data/CLEVR_v1.0/questions/CLEVR_test_questions.json', 
                       'data/CLEVR_v1.0/questions/CLEVR_train_questions.json']
    vocab_output_path = args.output
    for path in json_file_paths:
        json_data = load_json(path)

        if json_data is not None:
            for idx, question in enumerate(json_data.get('questions', [])):
                if 'program' in question:
                    for item in question['program']:
                        function = item.get('function', 'undefined_function')
                        if item.get('value_inputs'):
                            for value_input in item['value_inputs']:
                                key = f"{function}[{value_input}]"
                                if key not in program_vocab:
                                    program_vocab[key] = program_counter
                                    program_counter += 1
                        else:
                            key = function
                            if key not in program_vocab:
                                program_vocab[key] = program_counter
                                program_counter += 1
                if 'answer' in question:
                    if question['answer'] not in answer_vocab:
                        answer_vocab[question['answer']] = answer_counter
                        answer_counter += 1
                if 'question' in question:
                    tokens = tokenize(question['question'], add_end_token=False, add_start_token=False)
                    for word in tokens:
                        word = word.lower()
                        if word not in question_vocab:
                            question_vocab[word] = question_counter
                            question_counter += 1

    final_vocab = {"program_token_to_idx": program_vocab, "question_token_to_idx": question_vocab, "answer_token_to_idx": answer_vocab}
    try:
        with open(vocab_output_path, 'w', encoding='utf-8') as vocab_file:
            json.dump(final_vocab, vocab_file, indent=4)
        logging.info(f"Vocabulary successfully written to '{vocab_output_path}'.")
    except Exception as e:
        logging.error(f"An error occurred while writing the vocab to the file: {e}")

if __name__ == "__main__":
    main()
