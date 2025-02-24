import json
import os
from pprint import pprint
from typing import Dict, List, Any

def load_annotated_questions(file_path: str) -> Dict[str, Any]:
    """
    Loads the annotated_questions.json file.

    Args:
        file_path (str): Path to the annotated_questions.json file.

    Returns:
        Dict[str, Any]: Parsed JSON data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'questions' not in data:
        raise KeyError("The JSON file does not contain the 'questions' key.")
    
    return data

def display_first_question(data: Dict[str, Any]):
    """
    Displays the first question entry from the loaded data.

    Args:
        data (Dict[str, Any]): Parsed JSON data containing questions.
    """
    questions = data.get('questions', [])
    
    if not questions:
        print("No questions found in the dataset.")
        return
    
    first_question = questions[0]
    
    print("=== First Question Entry ===\n")
    
    # Display basic information
    print(f"Question Index: {first_question.get('question_index')}")
    print(f"Image Index: {first_question.get('image_index')}")
    print(f"Image Filename: {first_question.get('image_filename')}")
    print(f"Question Family Index: {first_question.get('question_family_index')}")
    print(f"Split: {first_question.get('split')}")
    print(f"Answer: {first_question.get('answer')}")
    print(f"Question: {first_question.get('question')}\n")
    
    # Display program steps with relevant objects
    annotated_program = first_question.get('annotated_program', [])
    
    if not annotated_program:
        print("No annotated program found for this question.")
        return
    
    print("Annotated Program Steps:\n")
    for idx, step in enumerate(annotated_program):
        print(f"Step {idx}:")
        print(f"  Function: {step.get('function')}")
        print(f"  Inputs: {step.get('inputs')}")
        print(f"  Value Inputs: {step.get('value_inputs')}")
        
        relevant_objects = step.get('relevant_objects', [])
        print(f"  Relevant Objects ({len(relevant_objects)}):")
        for obj_idx, obj in enumerate(relevant_objects):
            print(f"    Object {obj_idx}:")
            for attr, value in obj.items():
                print(f"      {attr}: {value}")
        print("\n")

def main():
    # Path to the annotated_questions.json file
    annotated_questions_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions.json"
    
    try:
        # Load the annotated questions
        data = load_annotated_questions(annotated_questions_path)
        
        # Display the first question entry
        display_first_question(data)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
