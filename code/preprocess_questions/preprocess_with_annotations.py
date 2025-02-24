import json
import os
import re
import h5py
import numpy as np

def build_vocab_and_data(json_path, numeric_range=(0, 1), decimals=3):
    """
    1) Loads the JSON file containing questions, each with 'image_index', 'question',
       'answer', 'annotated_program_string'.
    2) Builds a custom vocabulary for question tokens, answer tokens, and tokens
       appearing in the annotated program (including bounding box coords).
    3) Returns (questions_list, answers_list, annotated_program_list, image_indices,
       token_to_id, id_to_token).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    questions_data = data["questions"]

    # ------------------------------------
    # Step A: Prepare for numeric tokens
    # ------------------------------------
    # We'll store them in a dictionary: e.g. "0.000" -> some ID
    # We'll fill that in dynamically or we can pre-generate 0..1000 in steps of 0.001
    # to have "0.000", "0.001", ..., "1.000".
    numeric_tokens = {}
    dec_step = 1 / (10 ** decimals)  # e.g. 0.001
    start, end = numeric_range  # e.g. (0, 1)
    # Generate all coordinate strings
    # e.g. 0.000, 0.001, 0.002, ..., 1.000
    float_vals = []
    val = start
    # We'll do a while loop to ensure we get up to 1.000 inclusive
    while val <= end + 1e-9:
        coord_str = f"{val:.3f}"
        float_vals.append(coord_str)
        val += dec_step
    # numeric_tokens = { coord_str: ... } -> assigned later

    # We'll build a global set of tokens
    token_set = set()
    # We'll store everything in lists
    questions_list = []
    answers_list = []
    annotated_program_list = []
    image_indices = []

    # B) parse each question
    for qdict in questions_data:
        img_idx = qdict["image_index"]
        question_str = qdict["question"]
        answer_str = qdict["answer"]
        program_str = qdict["annotated_program_string"]

        image_indices.append(img_idx)

        # 1) parse question tokens
        # we can do a naive whitespace split
        q_tokens = question_str.strip().split()
        # add to token_set
        for t in q_tokens:
            token_set.add(t)
        questions_list.append(q_tokens)

        # 2) parse answer tokens (also naive)
        # some answers might be short like "yes", "no", or "2"
        a_tokens = answer_str.strip().split()
        for t in a_tokens:
            token_set.add(t)
        answers_list.append(a_tokens)

        # 3) parse annotated program
        # We'll parse or split by punctuation. e.g. "scene[]:(0.494,0.175) | ..."
        # We can do a simpler approach with a regex that captures bounding box floats
        # or tokens. We'll do the following approach:
        # - Split by whitespace and '|'
        # - Further split by punctuation if needed
        # We'll accumulate tokens in a program_tokens list

        prog_tokens = parse_annotated_program(program_str)
        # add to token_set
        for ptok in prog_tokens:
            # if it's in numeric_tokens range, we add numeric token
            # if it matches \d.\d or so. Let's handle that in parse_annotated_program.
            token_set.add(ptok)
        annotated_program_list.append(prog_tokens)

    # Now we have a global set of tokens that includes numeric strings and normal tokens.

    # Let's build token_to_id
    # We'll define a special <PAD>, <UNK> or so if needed
    special_tokens = ["<PAD>", "<UNK>"]
    # We'll sort the main tokens for consistency
    sorted_main_tokens = sorted(list(token_set))

    # final vocab
    # e.g. 0 -> <PAD>, 1 -> <UNK>, 2 -> first sorted token, ...
    token_to_id = {}
    idx = 0
    for st in special_tokens:
        token_to_id[st] = idx
        idx += 1

    for t in sorted_main_tokens:
        token_to_id[t] = idx
        idx += 1

    # build reverse map
    id_to_token = {v: k for k, v in token_to_id.items()}

    return (questions_list, answers_list, annotated_program_list, image_indices,
            token_to_id, id_to_token)

def parse_annotated_program(prog_str):
    """
    Splits the annotated_program_string into tokens.
    We'll:
     1) Replace '|' with spaces so we see them as separate.
     2) Then split on whitespace.
     3) For each chunk, break out punctuation like '(', ')', ':', ';', ',' except
        if it's a numeric coordinate like '0.494' we keep as is.
    """
    # Replace | with spaces
    line = prog_str.replace("|", " | ")
    chunks = line.split()

    all_tokens = []
    for ch in chunks:
        # If it's a bounding box coordinate e.g. (0.494,0.175,0.627,0.375)
        # we might separate out parentheses. We can do a small regex approach
        # or a manual parse. Let's do a simple approach:
        subparts = re.split(r'([\(\),:;])', ch)
        # re.split will keep delimiters in the list as separate items
        for sp in subparts:
            sp = sp.strip()
            if sp == "":
                continue
            # check if sp is a float with 3 decimals
            if re.match(r'^\d\.\d{3}$', sp) or re.match(r'^\d\.\d{3}$', sp) or re.match(r'^1\.\d{3}$', sp):
                # e.g. 0.494
                all_tokens.append(sp)
            else:
                # normal token or punctuation e.g. "(" or "scene[]:"
                all_tokens.append(sp)
    return all_tokens


def convert_to_ids(seq_list, token_to_id, max_len=30):
    """
    Convert each list of tokens in seq_list into a list of IDs, e.g. for question or answer.
    We'll also optionally pad to max_len. (You can handle variable length differently if you prefer.)
    Returns a 2D numpy array shape (len(seq_list), max_len).
    """
    res = []
    PAD_ID = token_to_id["<PAD>"]
    UNK_ID = token_to_id["<UNK>"]
    for tokens in seq_list:
        cur_ids = []
        for t in tokens:
            if t in token_to_id:
                cur_ids.append(token_to_id[t])
            else:
                cur_ids.append(UNK_ID)
        # pad or truncate
        cur_ids = cur_ids[:max_len]
        cur_ids += [PAD_ID]*(max_len - len(cur_ids))
        res.append(cur_ids)
    return np.array(res, dtype=np.int32)


def write_to_h5(h5_path, image_indices, q_ids, a_ids, p_ids):
    """
    Writes the arrays to HDF5:
     - image_index
     - question_tokens
     - answer_tokens
     - program_tokens
    """
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset("image_index", data=image_indices)
        f.create_dataset("question_tokens", data=q_ids)
        f.create_dataset("answer_tokens", data=a_ids)
        f.create_dataset("program_tokens", data=p_ids)

    print(f"[write_to_h5] wrote file {h5_path}")


def main():
    # Suppose your JSON is at "questions_annotated.json"
    json_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions.json"
    output_h5 = "mapped_sequences.h5"

    # 1) Build vocab & parse data
    (q_list, a_list, prog_list, img_indices,
     token_to_id, id_to_token) = build_vocab_and_data(json_path)

    # 2) Convert to IDs. For example, we choose some max_len for question/answer/program
    max_q_len = 20
    max_a_len = 5
    max_p_len = 100  # program can be longer

    q_ids = convert_to_ids(q_list, token_to_id, max_q_len)
    a_ids = convert_to_ids(a_list, token_to_id, max_a_len)
    p_ids = convert_to_ids(prog_list, token_to_id, max_p_len)

    # 3) Write to H5
    image_indices = np.array(img_indices, dtype=np.int32)
    write_to_h5(output_h5, image_indices, q_ids, a_ids, p_ids)

    # 4) (Optional) Save vocab to a JSON or text file so you can reuse
    vocab_out = "vocab.json"
    vocab_dict = {
        "token_to_id": token_to_id,
        "id_to_token": {str(i): t for i,t in id_to_token.items()}
    }
    with open(vocab_out, "w") as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"Saved vocab to {vocab_out}")


if __name__ == "__main__":
    main()
