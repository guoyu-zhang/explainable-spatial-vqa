import json

def load_json(file_path):
    """
    Loads a JSON file and returns its content as a Python variable.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data as a Python dictionary or list.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Successfully loaded JSON data from '{file_path}'.")
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def write_json(data, file_path):
    """
    Writes Python data to a JSON file.

    :param data: Python dictionary or list to write to JSON.
    :param file_path: Path to the output JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
            print(f"Successfully wrote new JSON data to '{file_path}'.")
    except Exception as e:
        print(f"Error writing to file '{file_path}': {e}")

def main():
    """
    Main function to load JSON data, extract the first element of 'b',
    and write the new JSON structure to a new file.
    """
    # Define the input and output file paths
    input_file_path = 'data/CLEVR_v1.0/questions/CLEVR_val_questions.json'
    output_file_path = 'CLEVR_val_questions_first.json'  # Change as needed

    # Load the original JSON data
    json_data = load_json(input_file_path)

    if json_data is not None:
        # Check if 'b' exists and is a list
        key_to_modify = 'questions'  # Replace 'questions' with 'b' if necessary
        if key_to_modify in json_data:
            if isinstance(json_data[key_to_modify], list):
                if len(json_data[key_to_modify]) > 0:
                    # Extract the first element
                    first_element = json_data[key_to_modify][0]

                    # Create the new JSON structure
                    new_json_data = json_data.copy()  # Shallow copy of the original data
                    new_json_data[key_to_modify] = [first_element]  # Replace 'b' with only the first element

                    # Optional: Print confirmation
                    print(f"Extracted the first element of '{key_to_modify}':")
                    print(json.dumps(first_element, indent=4))

                    # Write the new JSON data to the output file
                    write_json(new_json_data, output_file_path)
                else:
                    print(f"The list under key '{key_to_modify}' is empty. No elements to extract.")
            else:
                print(f"Error: The key '{key_to_modify}' is not associated with a list.")
        else:
            print(f"Error: The key '{key_to_modify}' does not exist in the JSON data.")

if __name__ == "__main__":
    main()
