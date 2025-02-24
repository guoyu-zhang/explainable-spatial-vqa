import os
import json

def stream_questions(filename, read_chunk_size=1024):
    """
    A generator that yields one question at a time from a JSON file with the following structure:
    
        {
          "questions": [ { ... }, { ... }, ... ]
        }
    
    This function reads the file incrementally in small chunks and uses JSONDecoder.raw_decode 
    to parse one question at a time.
    
    Args:
        filename (str): Path to the large JSON file.
        read_chunk_size (int): Number of bytes to read at a time.
        
    Yields:
        dict: A question object.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    with open(filename, 'r') as f:
        # Read until we find the beginning of the "questions" array.
        found_array = False
        while True:
            chunk = f.read(read_chunk_size)
            if not chunk:
                break
            buffer += chunk
            array_start = buffer.find('[')
            if array_start != -1:
                # Skip everything up to and including the '[' character.
                buffer = buffer[array_start+1:]
                found_array = True
                break

        if not found_array:
            raise ValueError("Could not find the start of the questions array in the file.")

        # Now, extract questions one by one from the buffer.
        while True:
            # Remove any leading whitespace or commas.
            buffer = buffer.lstrip(" \t\r\n,")
            # If we reach the end of the array, break.
            if buffer.startswith("]"):
                break
            try:
                # Attempt to decode the next JSON object.
                obj, idx = decoder.raw_decode(buffer)
                yield obj
                # Remove the parsed object from the buffer.
                buffer = buffer[idx:]
            except json.JSONDecodeError:
                # If not enough data is available, read another chunk.
                chunk = f.read(read_chunk_size)
                if not chunk:
                    break
                buffer += chunk

def split_json_file(input_path, output_dir, questions_per_file=10000):
    """
    Splits a large JSON file into multiple smaller JSON files.
    
    Each output file will have the structure:
        { "questions": [ ... ] }
    
    Args:
        input_path (str): Path to the large input JSON file.
        output_dir (str): Directory to write the split JSON files.
        questions_per_file (int): Number of questions per split file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    chunk = []
    file_index = 1
    total_questions = 0

    for question in stream_questions(input_path):
        chunk.append(question)
        total_questions += 1

        # If we've reached the desired number of questions, write out this chunk.
        if len(chunk) >= questions_per_file:
            output_path = os.path.join(output_dir, f"annotated_questions_part_{file_index}.json")
            with open(output_path, 'w') as out_f:
                json.dump({"questions": chunk}, out_f, indent=2)
            print(f"Wrote {len(chunk)} questions to {output_path}")
            file_index += 1
            chunk = []
    
    # Write out any remaining questions.
    if chunk:
        output_path = os.path.join(output_dir, f"annotated_questions_part_{file_index}.json")
        with open(output_path, 'w') as out_f:
            json.dump({"questions": chunk}, out_f, indent=2)
        print(f"Wrote {len(chunk)} questions to {output_path}")
    
    print(f"Finished splitting. Total questions processed: {total_questions}")

if __name__ == "__main__":
    # Path to your large JSON file.
    input_file = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions1.json"
    
    # Directory where the split files will be saved.
    output_directory = "split_json_files"
    
    # Number of questions per split file (adjust as needed).
    questions_per_file = 10000
    
    split_json_file(input_file, output_directory, questions_per_file)
