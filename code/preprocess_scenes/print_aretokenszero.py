import json

def inspect_token_samples(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    token_values = set()
    for q in data:
        for step in q.get("annotated_program", []):
            raw_out = step.get("output_values", "").strip()
            if '[' not in raw_out:  # token sample
                token_values.add(raw_out)
    print("Unique token raw_output values:", token_values)

if __name__ == "__main__":
    inspect_token_samples("/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions2.json")
