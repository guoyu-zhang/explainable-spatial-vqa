import json

def max_token_counts(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    max_input_tokens = 0
    max_output_tokens = 0
    
    for q in data:
        # Each question is expected to have an "annotated_program" field
        for step in q.get("annotated_program", []):
            input_vals = step.get("input_values", "").strip()
            output_vals = step.get("output_values", "").strip()
            # If the field contains a '[', we assume it's bounding box data and skip it.
            if '[' not in input_vals:
                tokens = input_vals.split()
                if len(tokens) > max_input_tokens:
                    max_input_tokens = len(tokens)
            if '[' not in output_vals:
                tokens = output_vals.split()
                if len(tokens) > max_output_tokens:
                    max_output_tokens = len(tokens)
    
    return max_input_tokens, max_output_tokens

if __name__ == "__main__":
    json_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions2.json"  # Change to your filename if needed
    max_in, max_out = max_token_counts(json_path)
    print("Maximum tokens in input_values (non-BBox):", max_in)
    print("Maximum tokens in output_values (non-BBox):", max_out)
