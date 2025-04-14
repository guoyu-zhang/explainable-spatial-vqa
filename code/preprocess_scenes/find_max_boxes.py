import json
import h5py
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Path to the HDF5 file containing the annotated questions (which is a JSON list)
ANNOTATED_QUESTIONS_H5 = "annotated_questions.h5"

def parse_bboxes(bbox_str: str):
    """
    Given a string containing bounding boxes (e.g. "[0.4939 0.1753 0.6269 0.3747] ..."),
    this function returns a list of bounding boxes as lists of 4 floats.
    """
    if not bbox_str or bbox_str.strip() == "":
        return []
    boxes = []
    # Find all groups of numbers inside square brackets
    matches = re.findall(r'\[([^\]]+)\]', bbox_str)
    for match in matches:
        # Split by whitespace and convert each to float
        numbers = [float(x) for x in match.strip().split()]
        if len(numbers) == 4:
            boxes.append(numbers)
    return boxes

def find_max_boxes():
    """
    Iterates over all annotated question objects (loaded from the HDF5 file)
    and finds the maximum number of bounding boxes in the "input_values" and 
    "output_values" fields in any program step.
    Returns:
        max_input_boxes: maximum number of input boxes found.
        max_output_boxes: maximum number of output boxes found.
    """
    with h5py.File(ANNOTATED_QUESTIONS_H5, 'r') as hf:
        data = hf['questions'][()]
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        annotated_questions = json.loads(data)
    
    max_input_boxes = 0
    max_output_boxes = 0
    for q in annotated_questions:
        for step in q.get("annotated_program", []):
            input_val = step.get("input_values", "")
            output_val = step.get("output_values", "")
            input_boxes = parse_bboxes(input_val)
            output_boxes = parse_bboxes(output_val)
            if len(input_boxes) > max_input_boxes:
                max_input_boxes = len(input_boxes)
            if len(output_boxes) > max_output_boxes:
                max_output_boxes = len(output_boxes)
    return max_input_boxes, max_output_boxes

def main():
    max_in, max_out = find_max_boxes()
    print(f"Maximum number of input bounding boxes in any sample: {max_in}")
    print(f"Maximum number of output bounding boxes in any sample: {max_out}")

if __name__ == "__main__":
    main()
