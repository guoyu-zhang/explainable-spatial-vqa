import json
from typing import List, Dict, Any
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


###############################################################################
# 1) Bounding box approximation
###############################################################################

def approximate_bounding_box(obj, scene):
    """
    Approximate bounding boxes based on the object's pixel_coords, 
    3d_coords, shape, etc. Returns (xmin, ymin, xmax, ymax) in [0,1].
    """
    x, y, _ = obj['pixel_coords']  # (x_pixel, y_pixel, depth)
    x3d, y3d, z3d = obj['3d_coords']
    
    rotation = scene['directions']['right']  # e.g. [cos_theta, sin_theta, 0]
    cos_theta, sin_theta, _ = rotation
    
    # Adjust (x3d, y3d) by rotation
    x1 = x3d * cos_theta + y3d * sin_theta
    y1 = x3d * (-sin_theta) + y3d * cos_theta
    
    # Baseline expansions
    height_d = 6.9 * z3d * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    # shape-specific logic: cylinder
    if obj['shape'] == 'cylinder':
        d = 9.4 + y1
        h = 6.4
        s = z3d
        height_u *= (s * (h/d + 1.0)) / ((s * (h/d + 1.0)) - (s * (h - s)/d))
        height_d = height_u * (h - s + d) / (h + s + d)
        width_l *= 11 / (10 + y1)
        width_r = width_l

    # shape-specific logic: cube
    if obj['shape'] == 'cube':
        scale_factor = 1.3 * 10.0 / (10.0 + y1)
        height_u *= scale_factor
        height_d = height_u
        width_l = height_u
        width_r = height_u

    # Convert to normalized coords (img: w=480, h=320)
    xmin = (x - width_l) / 480.0
    xmax = (x + width_r) / 480.0
    ymin = (y - height_d) / 320.0
    ymax = (y + height_u) / 320.0

    # Clip to [0,1]
    xmin = max(0.0, min(1.0, xmin))
    xmax = max(0.0, min(1.0, xmax))
    ymin = max(0.0, min(1.0, ymin))
    ymax = max(0.0, min(1.0, ymax))

    return (xmin, ymin, xmax, ymax)


###############################################################################
# 2) CLEVR functional program handlers
###############################################################################

def scene_handler(scene_struct, inputs, side_inputs):
    return list(range(len(scene_struct['objects'])))

def make_filter_handler(attribute):
    def filter_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 1
        value = side_inputs[0]
        return [idx for idx in inputs[0] if scene_struct['objects'][idx][attribute] == value]
    return filter_handler

def unique_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    if len(inputs[0]) != 1:
        return '__INVALID__'
    return inputs[0][0]

def relate_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
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
        assert len(inputs) == 1
        obj_idx = inputs[0]
        return scene_struct[f'_same_{attribute}'].get(obj_idx, [])
    return same_attr_handler

def make_query_handler(attribute):
    def query_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        obj_idx = inputs[0]
        val = scene_struct['objects'][obj_idx][attribute]
        if isinstance(val, list):
            if len(val) != 1:
                return '__INVALID__'
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
# 3) The main annotation function that produces a single string for each program
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


def annotate_questions_autoregressive_string(
    scenes_path: str,
    questions_path: str
) -> List[Dict]:
    """
    Returns a list of questions, each with a single string representation
    of the annotated program. We do NOT say "step0, step1" etc. We just
    produce lines like:

        scene[]:(0.123,0.456,0.789,0.111) ; (0.234,0.345,0.678,0.999) | filter_color[red]:(0.444,0.555,0.666,0.777)

    using:
      - " | " to separate steps
      - ":" to separate function name from bounding boxes
      - " ; " to separate multiple bounding boxes
      - "none" if no bounding boxes
      - bounding boxes are each (xmin,ymin,xmax,ymax), 3 decimals
    """
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
        relevant_objects_per_step = []
        node_outputs = []
        nodes_so_far = []

        # We'll build a single string for the entire program
        steps_str = []

        for idx, step in enumerate(program):
            node_type = step.get('function')
            if node_type is None:
                logging.error(f"Missing 'function' in step {idx}")
                relevant_objects_per_step.append([])
                node_outputs.append(None)
                # We'll store a placeholder in the string
                steps_str.append(f"missingFunction: none")
                continue

            inputs = step.get('inputs', [])
            side_inputs = step.get('value_inputs', [])

            node = {
                'type': node_type,
                'inputs': inputs,
                'side_inputs': side_inputs
            }
            nodes_so_far.append(node)

            try:
                # Execute
                temp_question = {'nodes': nodes_so_far}
                outputs = answer_question(temp_question, scene, cache_outputs=False)
                step_output = outputs[idx]

                # Determine relevant object indices
                if node_type == 'scene':
                    relevant_objs = list(range(len(scene['objects'])))

                elif (node_type.startswith('filter_') or node_type in ['relate','union','intersect']) \
                     or node_type.startswith('same_'):
                    if isinstance(step_output, list):
                        relevant_objs = step_output
                    else:
                        relevant_objs = []

                elif node_type == 'unique':
                    if isinstance(step_output, int):
                        relevant_objs = [step_output]
                    else:
                        relevant_objs = []

                elif node_type in [
                    'count','exist','greater_than','less_than',
                    'equal_color','equal_shape','equal_size',
                    'equal_material','equal_integer','equal_object'
                ]:
                    # returns int or bool
                    relevant_objs = []
                    for in_idx in inputs:
                        if 0 <= in_idx < len(relevant_objects_per_step):
                            relevant_objs.extend(relevant_objects_per_step[in_idx])
                    relevant_objs = list(set(relevant_objs))

                elif node_type.startswith('query_'):
                    # returns scalar
                    relevant_objs = []
                    for in_idx in inputs:
                        if 0 <= in_idx < len(relevant_objects_per_step):
                            relevant_objs.extend(relevant_objects_per_step[in_idx])
                    relevant_objs = list(set(relevant_objs))

                else:
                    logging.warning(f"Unhandled function '{node_type}' in step {idx}")
                    relevant_objs = []

                relevant_objects_per_step.append(relevant_objs)
                node_outputs.append(step_output)

                # Build a single-step string
                # e.g. "scene[]:(0.111,0.222,0.333,0.444) ; (0.555,0.666,0.777,0.888)"
                #
                # If function has side_inputs, e.g. filter_color[red]
                if side_inputs:
                    side_str = ",".join(str(s) for s in side_inputs)
                    step_label = f"{node_type}[{side_str}]"
                else:
                    step_label = f"{node_type}[]"

                # Now bounding boxes
                if len(relevant_objs) == 0:
                    # none
                    step_str = f"{step_label}:none"
                else:
                    bboxes_list = []
                    for obj_idx in relevant_objs:
                        if 0 <= obj_idx < len(scene['objects']):
                            (xmin, ymin, xmax, ymax) = approximate_bounding_box(scene['objects'][obj_idx], scene)
                            # Round to 3 decimals
                            bboxes_list.append(f"({round(xmin,3)},{round(ymin,3)},{round(xmax,3)},{round(ymax,3)})")
                    if bboxes_list:
                        joined_bboxes = " ; ".join(bboxes_list)
                        step_str = f"{step_label}:{joined_bboxes}"
                    else:
                        step_str = f"{step_label}:none"

                steps_str.append(step_str)

            except Exception as e:
                logging.error(f"Error in question_index {question.get('question_index')} step {idx}: {e}")
                relevant_objects_per_step.append([])
                node_outputs.append(None)
                steps_str.append(f"{node_type}[]:none")

        # Join all steps with " | "
        full_str = " | ".join(steps_str)

        # Store final
        annotated_question = question.copy()
        annotated_question['annotated_program_string'] = full_str
        annotated_questions.append(annotated_question)

    return annotated_questions


###############################################################################
# 4) main() usage example
###############################################################################

def main():
    scenes_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
    questions_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/questions/CLEVR_train_questions.json"
    output_path = "/Users/guoyuzhang/University/Y5/diss/vqa/code/data/CLEVR_v1.0/annotated_questions1.json"

    logging.info("Annotating questions into a single string per program...")
    ann_qs = annotate_questions_autoregressive_string(scenes_path, questions_path)

    with open(output_path, 'w') as f:
        json.dump({"questions": ann_qs}, f, indent=4)

    logging.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
