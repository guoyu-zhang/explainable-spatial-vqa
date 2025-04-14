import json
import re
import logging
import h5py

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"

def get_mismatch_stats():
    with h5py.File(ANNOTATED_QUESTIONS_H5, 'r') as hf:
        data = hf['questions'][()]
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        annotated_questions = json.loads(data)
    
    count_in_bbox_out_token = 0
    count_in_token_out_bbox = 0
    mismatch_examples = []  # store a few examples
    
    for q in annotated_questions:
        for step in q.get("annotated_program", []):
            # Skip samples where the function is 1 (handle both integer and string)
            func_val = step.get("function")
            if str(func_val).strip() == "1":
                continue

            input_val = step.get("input_values", "").strip()
            output_val = step.get("output_values", "").strip()
            # Determine type based on whether '[' is present
            input_is_bbox = '[' in input_val
            output_is_bbox = '[' in output_val
            
            if input_is_bbox and not output_is_bbox:
                count_in_bbox_out_token += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(("Input: Bounding boxes; Output: Token", step))
            elif (not input_is_bbox) and output_is_bbox:
                count_in_token_out_bbox += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(("Input: Token; Output: Bounding boxes", step))
    
    return count_in_bbox_out_token, count_in_token_out_bbox, mismatch_examples

def main():
    c1, c2, examples = get_mismatch_stats()
    logging.info(f"Cases with input as bounding boxes and output as token: {c1}")
    logging.info(f"Cases with input as token and output as bounding boxes: {c2}")
    
    if examples:
        logging.info("Examples of mismatches:")
        for mismatch_type, step in examples:
            logging.info(f"Mismatch type: {mismatch_type}")
            logging.info(f"Function: {step.get('function')}")
            logging.info(f"Input values: {step.get('input_values')}")
            logging.info(f"Output values: {step.get('output_values')}")
            logging.info("----")
    else:
        logging.info("No mismatches found.")

if __name__ == "__main__":
    main()
