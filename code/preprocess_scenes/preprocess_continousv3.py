import json
import re
from typing import List, Dict, Any
from collections import defaultdict
import logging
import h5py

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###############################################################################
# Bounding box logic (rounding to 4 decimal places)
###############################################################################
def approximate_bounding_box(obj, scene):
    x, y, depth = obj['pixel_coords']
    x3d, y3d, z3d = obj['3d_coords']
    
    rotation = scene['directions']['right']
    cos_theta, sin_theta, _ = rotation
    
    x1 = x3d * cos_theta + y3d * sin_theta
    y1 = x3d * (-sin_theta) + y3d * cos_theta
    
    height_d = 6.9 * z3d * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    if obj['shape'] == 'cylinder':
        d = 9.4 + y1
        h = 6.4
        s = z3d
        height_u *= (s * (h/d + 1.0)) / ((s * (h/d + 1.0)) - (s * (h - s)/d))
        height_d = height_u * (h - s + d) / (h + s + d)
        width_l *= 11 / (10 + y1)
        width_r = width_l

    if obj['shape'] == 'cube':
        scale_factor = 1.3 * 10.0 / (10.0 + y1)
        height_u *= scale_factor
        height_d = height_u
        width_l = height_u
        width_r = height_u

    xmin = max(0.0, min(1.0, (x - width_l) / 480.0))
    xmax = max(0.0, min(1.0, (x + width_r) / 480.0))
    ymin = max(0.0, min(1.0, (y - height_d) / 320.0))
    ymax = max(0.0, min(1.0, (y + height_u) / 320.0))

    return (round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4))

###############################################################################
# CLEVR function handlers
###############################################################################
def scene_handler(scene_struct, inputs, side_inputs):
    return list(range(len(scene_struct['objects'])))

def make_filter_handler(attribute):
    def filter_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1, f"filter_{attribute} expects 1 input"
        assert len(side_inputs) == 1, f"filter_{attribute} expects 1 side_input"
        value = side_inputs[0]
        return [idx for idx in inputs[0] if scene_struct['objects'][idx][attribute] == value]
    return filter_handler

def unique_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1, "unique expects 1 input"
    if len(inputs[0]) != 1:
        return '__INVALID__'
    return inputs[0][0]

def relate_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1, "relate expects 1 input"
    assert len(side_inputs) == 1, "relate expects 1 side_input"
    relation = side_inputs[0]
    return scene_struct['relationships'].get(relation, {}).get(inputs[0], [])

def union_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    return sorted(list(set(inputs[0]) | set(inputs[1])))

def intersect_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    return sorted(list(set(inputs[0]) & set(inputs[1])))

def count_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    return len(inputs[0])

def make_same_attr_handler(attribute):
    def same_attr_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1, f"same_{attribute} expects 1 input"
        obj_idx = inputs[0]
        return scene_struct[f'_same_{attribute}'].get(obj_idx, [])
    return same_attr_handler

def make_query_handler(attribute):
    def query_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        obj_idx = inputs[0]
        val = scene_struct['objects'][obj_idx][attribute]
        if isinstance(val, list) and len(val) != 1:
            return '__INVALID__'
        if isinstance(val, list) and len(val) == 1:
            return val[0]
        return val
    return query_handler

def exist_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    return len(inputs[0]) > 0

def equal_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    return inputs[0] == inputs[1]

def less_than_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    return inputs[0] < inputs[1]

def greater_than_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    return inputs[0] > inputs[1]

execute_handlers = {
    'scene': scene_handler,
    'filter_color': make_filter_handler('color'),
    'filter_shape': make_filter_handler('shape'),
    'filter_material': make_filter_handler('material'),
    'filter_size': make_filter_handler('size'),
    'filter_objectcategory': make_filter_handler('objectcategory'),
    'unique': unique_handler,
    'relate': relate_handler,
    'union': union_handler,
    'intersect': intersect_handler,
    'count': count_handler,
    'query_color': make_query_handler('color'),
    'query_shape': make_query_handler('shape'),
    'query_material': make_query_handler('material'),
    'query_size': make_query_handler('size'),
    'exist': exist_handler,
    'equal_color': equal_handler,
    'equal_shape': equal_handler,
    'equal_integer': equal_handler,
    'equal_material': equal_handler,
    'equal_size': equal_handler,
    'equal_object': equal_handler,
    'less_than': less_than_handler,
    'greater_than': greater_than_handler,
    'same_color': make_same_attr_handler('color'),
    'same_shape': make_same_attr_handler('shape'),
    'same_size': make_same_attr_handler('size'),
    'same_material': make_same_attr_handler('material'),
}

###############################################################################
# Answering questions using the program
###############################################################################
def answer_question(question: Dict[str, Any], scene_struct: Dict[str, Any], cache_outputs: bool = True) -> List[Any]:
    node_outputs = []
    for node in question['nodes']:
        if cache_outputs and '_output' in node:
            node_output = node['_output']
        else:
            node_type = node['type']
            if node_type not in execute_handlers:
                raise ValueError(f"Unknown function type: {node_type}")
            handler = execute_handlers[node_type]
            node_inputs = [node_outputs[idx] for idx in node['inputs']]
            side_inputs = node.get('side_inputs', [])
            node_output = handler(scene_struct, node_inputs, side_inputs)
            if cache_outputs:
                node['_output'] = node_output
        node_outputs.append(node_output)
        if node_output == '__INVALID__':
            break
    return node_outputs

###############################################################################
# Loading scenes and questions
###############################################################################
def load_scenes(scenes_path: str) -> Dict[int, Dict[str, Any]]:
    with open(scenes_path, 'r') as f:
        scenes_data = json.load(f)
    scenes = {}
    for scene in scenes_data['scenes']:
        preprocess_scene_relationships(scene)
        scenes[scene['image_index']] = scene
    return scenes

def load_questions(questions_path: str) -> List[Dict[str, Any]]:
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)
    return questions_data['questions']

def preprocess_scene_relationships(scene_struct: Dict[str, Any]):
    relationships = scene_struct.get('relationships', {})
    processed_relationships = {}
    for relation, rel_list in relationships.items():
        processed_relationships[relation] = defaultdict(list)
        for subject_idx, related_objs in enumerate(rel_list):
            for object_idx in related_objs:
                processed_relationships[relation][subject_idx].append(object_idx)
    scene_struct['relationships'] = processed_relationships
    for attr in ['color', 'shape', 'size', 'material']:
        same_attr = {}
        for i, obj1 in enumerate(scene_struct['objects']):
            matches = []
            for j, obj2 in enumerate(scene_struct['objects']):
                if i != j and obj1[attr] == obj2[attr]:
                    matches.append(j)
            same_attr[i] = matches
        scene_struct[f'_same_{attr}'] = same_attr

###############################################################################
# Vocabulary conversion functions
###############################################################################
# Define a special token for empty fields.
EMPTY_TOKEN = "<EMPTY>"

def canonicalize(token: str) -> str:
    token_lower = token.lower()
    if token_lower in {"yes", "true"}:
        return "true"
    if token_lower in {"no", "false"}:
        return "false"
    return token

def tokenize_field(text: str, field: str) -> List[str]:
    if field == "function":
        return [text] if text else []
    return re.findall(r'\[|\]|[^\[\]\s]+', text)

BBOX_PATTERN = re.compile(r'\[\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\]')
def is_bounding_box_text(text: str) -> bool:
    matches = BBOX_PATTERN.findall(text)
    if not matches:
        return False
    reconstructed = " ".join(matches)
    return reconstructed.strip() == text.strip()

# --- Modified vocabulary building using separate vocabs ---
def build_vocab_from_dataset(annotated_qs: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    vocab_function = {}
    vocab_other = {}
    next_index_function = 0
    next_index_other = 0

    def add_tokens_from_text(text: str, field: str):
        nonlocal next_index_function, next_index_other
        if is_bounding_box_text(text):
            return
        tokens = tokenize_field(text, field)
        for token in tokens:
            can_tok = canonicalize(token)
            if field == "function":
                if can_tok not in vocab_function:
                    vocab_function[can_tok] = next_index_function
                    next_index_function += 1
            else:
                if can_tok not in vocab_other:
                    vocab_other[can_tok] = next_index_other
                    next_index_other += 1

    for annotated_q in annotated_qs:
        add_tokens_from_text(annotated_q.get("answer", ""), "other")
        # For final chain-of-thought, only add the function part (first token)
        for chain in annotated_q.get("final_chain_of_thought", []):
            parts = chain.split(maxsplit=1)
            add_tokens_from_text(parts[0], "function")
        for step in annotated_q.get("annotated_program", []):
            add_tokens_from_text(step.get("function", ""), "function")
            add_tokens_from_text(step.get("input_values", ""), "other")
            add_tokens_from_text(step.get("output_values", ""), "other")
    # Ensure the empty token is in the "other" vocabulary.
    if EMPTY_TOKEN not in vocab_other:
        vocab_other[EMPTY_TOKEN] = next_index_other
        next_index_other += 1

    return {"function": vocab_function, "other": vocab_other}

# --- Modified apply_vocab to use separate vocabs and handle empty fields ---
def apply_vocab(annotated_q: Dict[str, Any], vocabs: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    vocab_function = vocabs["function"]
    vocab_other = vocabs["other"]

    def convert_text(text: str, field: str) -> str:
        tokens = tokenize_field(text, field)
        indices = []
        for token in tokens:
            can_tok = canonicalize(token)
            # For "other", do not convert tokens that represent numbers.
            if field == "other" and token.replace('.', '', 1).isdigit():
                indices.append(token)
            else:
                if field == "function":
                    if can_tok in vocab_function:
                        indices.append(str(vocab_function[can_tok]))
                else:
                    if can_tok in vocab_other:
                        indices.append(str(vocab_other[can_tok]))
        return " ".join(indices)
    
    annotated_q["answer"] = convert_text(annotated_q.get("answer", ""), "other")
    
    def convert_chain(chain: str) -> str:
        parts = chain.split(maxsplit=1)
        func_part = parts[0]
        # Convert the function part only
        converted_func = convert_text(func_part, "function")
        # Leave the second part unchanged
        rest = parts[1] if len(parts) > 1 else ""
        return f"{converted_func} {rest}".strip() if rest else converted_func

    annotated_q["final_chain_of_thought"] = [convert_chain(chain) for chain in annotated_q.get("final_chain_of_thought", [])]
    
    for step in annotated_q.get("annotated_program", []):
        step["function"] = convert_text(step.get("function", ""), "function")
        
        input_val = step.get("input_values", "")
        if is_bounding_box_text(input_val):
            step["input_values"] = input_val
        else:
            converted_input = convert_text(input_val, "other")
            # If conversion is empty, use the EMPTY_TOKEN conversion.
            if not converted_input.strip():
                converted_input = convert_text(EMPTY_TOKEN, "other")
            step["input_values"] = converted_input
        
        output_val = step.get("output_values", "")
        if is_bounding_box_text(output_val):
            step["output_values"] = output_val
        else:
            converted_output = convert_text(output_val, "other")
            # If conversion is empty, use the EMPTY_TOKEN conversion.
            if not converted_output.strip():
                converted_output = convert_text(EMPTY_TOKEN, "other")
            step["output_values"] = converted_output
    
    return annotated_q

###############################################################################
# Annotation function
###############################################################################
SPATIAL_FUNCTIONS = {
    "scene", "filter_color", "filter_shape", "filter_material", "filter_size", "filter_objectcategory", 
    "relate", "union", "intersect", "unique", "same_color", "same_shape", "same_size", "same_material"
}
NON_SPATIAL_FUNCTIONS = {
    "count", "exist", "query_color", "query_shape", "query_material", "query_size",
    "equal_integer", "less_than", "greater_than", "equal_color", "equal_shape", "equal_size",
    "equal_material", "equal_object"
}

def annotate_questions_with_relevant_objects(scenes_path: str, questions_path: str) -> List[Dict]:
    scenes = load_scenes(scenes_path)
    questions = load_questions(questions_path)
    
    annotated_questions = []
    for question in questions:
        image_index = question['image_index']
        program = question['program']

        if image_index not in scenes:
            logging.warning(f"Scene with image_index {image_index} not found.")
            continue

        scene = scenes[image_index]
        scene_struct = scene

        relevant_objects_per_step = []
        node_outputs = []
        nodes_so_far = []

        for idx, step in enumerate(program):
            node_type = step.get('function')
            if node_type is None:
                logging.error(f"Missing 'function' key in program step {idx}")
                relevant_objects_per_step.append([])
                node_outputs.append(None)
                continue

            inputs = step.get('inputs', [])
            value_inputs = step.get('value_inputs', [])
            node = {
                'type': node_type,
                'inputs': inputs,
                'side_inputs': value_inputs
            }
            nodes_so_far.append(node)

            try:
                temp_question = {'nodes': nodes_so_far}
                outputs = answer_question(temp_question, scene_struct, cache_outputs=False)
                step_output = outputs[idx]

                if node_type == 'scene':
                    relevant_objs = list(range(len(scene_struct['objects'])))
                elif (node_type.startswith('filter_') or node_type in ['relate', 'union', 'intersect']) or node_type.startswith('same_'):
                    relevant_objs = step_output if isinstance(step_output, list) else []
                elif node_type == 'unique':
                    relevant_objs = [step_output] if isinstance(step_output, int) else []
                elif node_type in NON_SPATIAL_FUNCTIONS or node_type.startswith('query_'):
                    relevant_objs = []
                else:
                    logging.warning(f"Unhandled function '{node_type}' in step {idx}")
                    relevant_objs = []

                relevant_objects_per_step.append(relevant_objs)
                node_outputs.append(step_output)

            except Exception as e:
                logging.error(f"Error at program step {idx}: {e}")
                relevant_objects_per_step.append([])
                node_outputs.append(None)

        annotated_program = []
        chain_list = []
        for i, step in enumerate(program):
            annotated_step = step.copy()
            annotated_step.pop('value_inputs', None)

            function_name = annotated_step.get('function', '')
            if 'value_inputs' in step and step['value_inputs']:
                combined_function = f"{function_name}[{','.join(map(str, step['value_inputs']))}]"
            else:
                combined_function = function_name
            annotated_step['function'] = combined_function

            # Use the output_values of the input steps as the input_values for the current step.
            input_values_list = []
            for inp in step.get('inputs', []):
                if inp < len(annotated_program):
                    input_values_list.append(annotated_program[inp]['output_values'])
                else:
                    input_values_list.append(str(node_outputs[inp]))
            annotated_step['input_values'] = " ".join(input_values_list).strip()

            chain_elem = f"{combined_function} " + " ".join(map(str, step.get('inputs', [])))
            chain_list.append(chain_elem.strip())

            base_function = combined_function.split('[')[0]
            if base_function in NON_SPATIAL_FUNCTIONS:
                output_value = str(node_outputs[i])
                if output_value.startswith("[") and output_value.endswith("]"):
                    output_value = output_value[1:-1]
                annotated_step['output_values'] = output_value.strip()
            elif base_function in SPATIAL_FUNCTIONS:
                bboxes = []
                rel_objs = relevant_objects_per_step[i] if i < len(relevant_objects_per_step) else []
                for obj_idx in rel_objs:
                    if obj_idx is not None and 0 <= obj_idx < len(scene['objects']):
                        bbox = approximate_bounding_box(scene['objects'][obj_idx], scene)
                        bbox_str = f"[{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}]"
                        bboxes.append(bbox_str)
                annotated_step['output_values'] = " ".join(bboxes).strip()
            else:
                annotated_step['output_values'] = ""
            annotated_program.append(annotated_step)

        annotated_question = question.copy()
        for key in ["program", "image_filename", "split", "question_family_index"]:
            annotated_question.pop(key, None)
        annotated_question['annotated_program'] = annotated_program
        annotated_question['final_chain_of_thought'] = chain_list.copy()
        annotated_questions.append(annotated_question)

    return annotated_questions

###############################################################################
# Main
###############################################################################
def main():
    scenes_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
    questions_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/questions/CLEVR_train_questions.json"
    
    json_output_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions3.json"
    vocab_output_path = "vocab3.json"
    
    logging.info("Loading scenes and questions, then annotating for the whole dataset...")
    annotated_questions = annotate_questions_with_relevant_objects(scenes_path, questions_path)
    
    if not annotated_questions:
        logging.error("No annotated questions found.")
        return

    # Build separate vocabularies from the annotated questions (non bounding box tokens only)
    vocabs = build_vocab_from_dataset(annotated_questions)
    # Apply the vocabulary conversion to the annotated questions using separate vocabs for functions and other tokens
    converted_questions = [apply_vocab(q, vocabs) for q in annotated_questions]
    
    # Save the converted annotated questions to a JSON file
    with open(json_output_path, 'w') as f:
        json.dump(converted_questions, f, indent=4)
    logging.info(f"Saved converted annotated questions to {json_output_path}")
    
    # Save the vocabularies to a JSON file
    with open(vocab_output_path, 'w') as f:
        json.dump(vocabs, f, indent=4)
    logging.info(f"Saved vocabularies to {vocab_output_path}")
    
    # Save the questions JSON into an HDF5 file
    dt = h5py.string_dtype(encoding='utf-8')
    h5_questions_path = "annotated_questions.h5"
    with h5py.File(h5_questions_path, 'w') as hf:
        for i, q in enumerate(converted_questions):
            q_str = json.dumps(q)
            hf.create_dataset(f"q_{i}", data=q_str, dtype=dt)


if __name__ == "__main__":
    main()
