import json
from typing import List, Dict, Any
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

###############################################################################
# Bounding box logic (modified to round values to 1 decimal place)
###############################################################################

def approximate_bounding_box(obj, scene):
    """
    Approximate bounding boxes based on the object's pixel_coords, 
    3d_coords, shape, etc. Returns (xmin, ymin, xmax, ymax) in [0,1],
    with each value rounded to 1 decimal place.
    """
    # Extract values
    x, y, depth = obj['pixel_coords']  # pixel coords [x_pixel, y_pixel, depth]
    x3d, y3d, z3d = obj['3d_coords']
    
    # Rotation vector from the scene
    rotation = scene['directions']['right']  # e.g., [cos_theta, sin_theta, 0]
    cos_theta, sin_theta, _ = rotation
    
    # Adjust (x3d, y3d) by the rotation:
    x1 = x3d * cos_theta + y3d * sin_theta
    y1 = x3d * (-sin_theta) + y3d * cos_theta
    
    # Baseline bounding-box expansions
    height_d = 6.9 * z3d * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    # Shape-specific logic: cylinder
    if obj['shape'] == 'cylinder':
        d = 9.4 + y1
        h = 6.4
        s = z3d
        height_u *= (s * (h/d + 1.0)) / ((s * (h/d + 1.0)) - (s * (h - s)/d))
        height_d = height_u * (h - s + d) / (h + s + d)
        width_l *= 11 / (10 + y1)
        width_r = width_l

    # Shape-specific logic: cube
    if obj['shape'] == 'cube':
        scale_factor = 1.3 * 10.0 / (10.0 + y1)
        height_u *= scale_factor
        height_d = height_u
        width_l = height_u
        width_r = height_u

    # Convert to normalized image coords using image dims: height=320, width=480
    xmin = (x - width_l) / 480.0
    xmax = (x + width_r) / 480.0
    ymin = (y - height_d) / 320.0
    ymax = (y + height_u) / 320.0

    # Clip to [0,1]
    xmin = max(0.0, min(1.0, xmin))
    xmax = max(0.0, min(1.0, xmax))
    ymin = max(0.0, min(1.0, ymin))
    ymax = max(0.0, min(1.0, ymax))
    
    # Round each coordinate to 1 decimal place
    xmin = round(xmin, 1)
    xmax = round(xmax, 1)
    ymin = round(ymin, 1)
    ymax = round(ymax, 1)

    return (xmin, ymin, xmax, ymax)

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

    # Precompute same attributes for 'same_*' functions
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
# Annotation function
###############################################################################

# Define sets of spatial and non-spatial function base names.
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
    
    # Process all examples (remove slicing [:1])
    annotated_questions = []
    for question in questions:
        image_index = question['image_index']
        program = question['program']

        if image_index not in scenes:
            logging.warning(f"Scene with image_index {image_index} not found.")
            continue

        scene = scenes[image_index]
        scene_struct = scene

        relevant_objects_per_step = []  # For spatial functions: list of object indices.
        node_outputs = []              # For all functions: the actual node outputs.
        nodes_so_far = []

        # Process the program steps to compute outputs.
        for idx, step in enumerate(program):
            node_type = step.get('function')
            if node_type is None:
                logging.error(f"Missing 'function' key in program step {idx} of question {question.get('question_index', 'N/A')}")
                relevant_objects_per_step.append([])
                node_outputs.append(None)
                continue

            inputs = step.get('inputs', [])
            value_inputs = step.get('value_inputs', [])

            # Build a CLEVR node.
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

                # Determine the set of spatial output indices.
                if node_type == 'scene':
                    relevant_objs = list(range(len(scene_struct['objects'])))
                elif (node_type.startswith('filter_') or node_type in ['relate', 'union', 'intersect']) or node_type.startswith('same_'):
                    relevant_objs = step_output if isinstance(step_output, list) else []
                elif node_type == 'unique':
                    relevant_objs = [step_output] if isinstance(step_output, int) else []
                elif node_type in NON_SPATIAL_FUNCTIONS or node_type.startswith('query_'):
                    relevant_objs = []  # Non-spatial functions.
                else:
                    logging.warning(f"Unhandled function '{node_type}' in step {idx} of question {question.get('question_index', 'N/A')}")
                    relevant_objs = []

                relevant_objects_per_step.append(relevant_objs)
                node_outputs.append(step_output)

            except Exception as e:
                logging.error(f"Error processing question {question.get('question_index', 'N/A')} at program step {idx}: {e}")
                relevant_objects_per_step.append([])
                node_outputs.append(None)

        # Build the annotated program.
        annotated_program = []
        chain_list = []  # To accumulate the chain_of_thought.
        for i, step in enumerate(program):
            annotated_step = step.copy()

            # Combine "function" and "value_inputs" into one string.
            function_name = annotated_step.get('function', '')
            value_inputs = annotated_step.get('value_inputs', [])
            if value_inputs:
                combined_function = f"{function_name}[{','.join(map(str, value_inputs))}]"
            else:
                combined_function = function_name
            annotated_step['function'] = combined_function

            # Update chain_of_thought.
            chain_list.append(combined_function)
            annotated_step['chain_of_thought'] = chain_list.copy()

            # Determine the base function name.
            base_function = combined_function.split('[')[0]

            # Build "input_values":
            if base_function in NON_SPATIAL_FUNCTIONS:
                # For non-spatial functions, use the actual node outputs of the referenced input steps.
                input_values = [node_outputs[inp] for inp in step.get('inputs', [])]
            else:
                # For spatial functions, compute bounding boxes.
                input_values = []
                for inp in step.get('inputs', []):
                    if inp < len(relevant_objects_per_step):
                        prev_obj_indices = relevant_objects_per_step[inp]
                        for obj_idx in prev_obj_indices:
                            if obj_idx is not None and 0 <= obj_idx < len(scene['objects']):
                                bbox = approximate_bounding_box(scene['objects'][obj_idx], scene)
                                input_values.append({'bbox': bbox})
            annotated_step['input_values'] = input_values

            # Build "output_values":
            if base_function in NON_SPATIAL_FUNCTIONS:
                annotated_step['output_values'] = node_outputs[i]
            elif base_function in SPATIAL_FUNCTIONS:
                output_boxes = []
                rel_objs = relevant_objects_per_step[i] if i < len(relevant_objects_per_step) else []
                for obj_idx in rel_objs:
                    if obj_idx is not None and 0 <= obj_idx < len(scene['objects']):
                        bbox = approximate_bounding_box(scene['objects'][obj_idx], scene)
                        output_boxes.append({'bbox': bbox})
                annotated_step['output_values'] = output_boxes
            else:
                annotated_step['output_values'] = []

            annotated_program.append(annotated_step)

        # Add an extra "end" step at the end of the annotated program.
        if annotated_program:
            last_step = annotated_program[-1]
            end_step = {
                "inputs": [len(annotated_program) - 1],  # Reference the last step.
                "function": "end",
                "value_inputs": [],
                "chain_of_thought": chain_list.copy() + ["end"],
                # "input_values" will be a copy of the last step's output_values.
                "input_values": last_step.get("output_values", []),
                # "output_values" is set to the answer from the question.
                "output_values": question.get("answer")
            }
            annotated_program.append(end_step)

        annotated_question = question.copy()
        annotated_question['annotated_program'] = annotated_program
        annotated_questions.append(annotated_question)

    return annotated_questions

###############################################################################
# Main
###############################################################################

def main():
    scenes_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
    questions_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/questions/CLEVR_train_questions.json"
    output_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions1.json"

    logging.info("Loading scenes and questions, then annotating with bounding boxes...")
    annotated_questions = annotate_questions_with_relevant_objects(scenes_path, questions_path)

    with open(output_path, 'w') as f:
        json.dump({"questions": annotated_questions}, f, indent=4)
    logging.info(f"Saved annotated questions (with bounding boxes) to {output_path}")

if __name__ == "__main__":
    main()
