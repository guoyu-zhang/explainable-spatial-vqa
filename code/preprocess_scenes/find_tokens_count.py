import json
import re
import logging
import h5py

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"

def get_non_bbox_stats():
    # Open the annotated_questions.h5 file and load the JSON list.
    with h5py.File(ANNOTATED_QUESTIONS_H5, 'r') as hf:
        data = hf['questions'][()]
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        annotated_questions = json.loads(data)
    
    max_input_tokens = 0
    max_output_tokens = 0
    input_lengths = []
    output_lengths = []
    count_input = 0
    count_output = 0

    # Iterate over all questions and their program steps.
    for q in annotated_questions:
        for step in q.get("annotated_program", []):
            input_val = step.get("input_values", "").strip()
            output_val = step.get("output_values", "").strip()
            # For input values that are not bounding boxes:
            if '[' not in input_val and input_val != "":
                tokens = input_val.split()
                token_count = len(tokens)
                input_lengths.append(token_count)
                count_input += 1
                if token_count > max_input_tokens:
                    max_input_tokens = token_count
            # For output values that are not bounding boxes:
            if '[' not in output_val and output_val != "":
                tokens = output_val.split()
                token_count = len(tokens)
                output_lengths.append(token_count)
                count_output += 1
                if token_count > max_output_tokens:
                    max_output_tokens = token_count

    avg_input_tokens = sum(input_lengths) / len(input_lengths) if input_lengths else 0
    avg_output_tokens = sum(output_lengths) / len(output_lengths) if output_lengths else 0

    return {
        "max_input_tokens": max_input_tokens,
        "avg_input_tokens": avg_input_tokens,
        "count_input_non_bbox": count_input,
        "max_output_tokens": max_output_tokens,
        "avg_output_tokens": avg_output_tokens,
        "count_output_non_bbox": count_output
    }

def main():
    stats = get_non_bbox_stats()
    logging.info("Statistics for non-bounding box examples:")
    logging.info(f"Max number of tokens in input_values: {stats['max_input_tokens']}")
    logging.info(f"Average tokens in input_values: {stats['avg_input_tokens']:.2f}")
    logging.info(f"Count of non-bbox input_values examples: {stats['count_input_non_bbox']}")
    logging.info(f"Max number of tokens in output_values: {stats['max_output_tokens']}")
    logging.info(f"Average tokens in output_values: {stats['avg_output_tokens']:.2f}")
    logging.info(f"Count of non-bbox output_values examples: {stats['count_output_non_bbox']}")

if __name__ == "__main__":
    main()
