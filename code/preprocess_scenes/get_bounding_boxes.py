#!/usr/bin/env python3
"""
Example script to:
  1) Read a CLEVR scenes JSON file.
  2) Compute approximate bounding boxes for each object (in normalized coords).
  3) Store bounding boxes + class labels in an .h5 file.
"""

import json
import numpy as np
import h5py
import os

# -------------------------------------------------------------------------
# Adjust these as needed:
JSON_SCENES_PATH = "../data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
H5_OUTPUT_PATH   = "../h5_files/train_scenes.h5"
# -------------------------------------------------------------------------

def generate_label_map():
    """
    Returns a list of all possible combinations of 
    [size, color, material, shape].
    
    Also returns a dictionary 'label_to_id' mapping 'large gray rubber cube'
    -> integer ID, etc.
    """
    sizes = ['large', 'small']
    colors = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
    materials = ['rubber', 'metal']
    shapes = ['cube', 'sphere', 'cylinder']
    
    names = [
        f"{s} {c} {m} {sh}"
        for s in sizes
        for c in colors
        for m in materials
        for sh in shapes
    ]
    
    # Sort for consistency
    names = sorted(names)
    label_to_id = {name: i+1 for i, name in enumerate(names)}

    return names, label_to_id


def approximate_bounding_box(obj, scene):
    """
    Approximate bounding boxes based on the object's pixel_coords, 
    3d_coords, shape, etc.

    This logic is adapted from your TensorFlow script, which:
     - Takes the scene['directions']['right'] as a rotation vector.
     - Manipulates width/height expansions differently for cubes, cylinders, etc.
     - Normalizes the bounding boxes by image width=480, height=320.

    Returns (xmin, ymin, xmax, ymax) in [0,1].
    """
    # Extract values
    x, y, z = obj['pixel_coords']  # pixel coords [x_pixel, y_pixel, depth]
    x3d, y3d, z3d = obj['3d_coords']
    
    # rotation vector from the scene
    rotation = scene['directions']['right']
    cos_theta, sin_theta, _ = rotation
    
    # The TF code adjusts (x3d, y3d) by the rotation:
    #   x1 = x1 * cos_theta + y1 * sin_theta
    #   y1 = x1 * -sin_theta + y1 * cos_theta
    # WARNING: watch for overwriting x1
    x1 = x3d * cos_theta + y3d * sin_theta
    y1 = x3d * (-sin_theta) + y3d * cos_theta
    
    # Baseline bounding-box expansions
    # (From the TF code, it starts with height_d = 6.9 * z3d * (15 - y1)/2.0)
    height_d = 6.9 * z3d * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    # Then special logic if shape is cylinder:
    #   if obj['shape'] == 'cylinder':
    #       ...
    if obj['shape'] == 'cylinder':
        d = 9.4 + y1
        h = 6.4
        s = z3d
        # from the original code:
        # height_u *= (s*(h/d + 1)) / ((s*(h/d + 1)) - (s*(h-s)/d))
        # height_d = height_u * (h-s+d)/(h + s + d)
        height_u *= (s * (h/d + 1.0)) / ((s * (h/d + 1.0)) - (s * (h - s)/d))
        height_d = height_u * (h - s + d) / (h + s + d)
        
        width_l *= 11/(10 + y1)
        width_r = width_l

    # If shape is cube:
    if obj['shape'] == 'cube':
        # from the code:
        # height_u *= 1.3 * 10 / (10 + y1)
        # height_d = height_u
        # width_l = height_u
        # width_r = height_u
        scale_factor = 1.3 * 10.0 / (10.0 + y1)
        height_u *= scale_factor
        height_d = height_u
        width_l = height_u
        width_r = height_u

    # Spheres didn't have special logic in the original code,
    # so they keep the default expansions.

    # Convert to normalized image coords
    # The TF code used image dims: height=320, width=480
    # (y = 0..320, x = 0..480)
    xmin = (x - width_l) / 480.0
    xmax = (x + width_r) / 480.0
    ymin = (y - height_d) / 320.0
    ymax = (y + height_u) / 320.0

    # Clip to [0,1] so we don't get negative or >1
    xmin = max(0.0, min(1.0, xmin))
    xmax = max(0.0, min(1.0, xmax))
    ymin = max(0.0, min(1.0, ymin))
    ymax = max(0.0, min(1.0, ymax))

    return (xmin, ymin, xmax, ymax)


def main():
    # 1) Load scenes JSON
    with open(JSON_SCENES_PATH, "r") as f:
        scene_data = json.load(f)
    scenes = scene_data["scenes"]
    num_scenes = len(scenes)
    print(f"Loaded {num_scenes} scenes from {JSON_SCENES_PATH}")

    # 2) Generate label map (for all combos of size/color/material/shape)
    label_names, label_to_id = generate_label_map()
    print(f"Generated {len(label_to_id)} label IDs.")

    # 3) Find max number of objects (for fixed-size arrays)
    max_objects = max(len(scene["objects"]) for scene in scenes)
    print(f"Max objects in any scene: {max_objects}")

    # 4) Prepare arrays
    # We'll store bounding_boxes in shape (num_scenes, max_objects, 4)
    #   each row: [xmin, ymin, xmax, ymax]
    # We'll also store class_labels in shape (num_scenes, max_objects)
    # We'll store image_index in shape (num_scenes,)
    # We'll store image_filename strings (variable length) separately or as bytes.
    bounding_boxes = np.zeros((num_scenes, max_objects, 4), dtype=np.float32)
    class_labels   = np.zeros((num_scenes, max_objects), dtype=np.int32)
    image_index    = np.zeros((num_scenes,), dtype=np.int32)
    
    # Let's store filenames in a list of strings. 
    # Later we can write them as variable-length dataset or just skip them.
    image_filenames = []

    # 5) Fill the data
    for i, scene in enumerate(scenes):
        image_index[i] = scene["image_index"]
        image_filenames.append(scene["image_filename"].encode("utf8"))
        
        objs = scene["objects"]
        for j, obj in enumerate(objs):
            # 5a) bounding box
            (xmin, ymin, xmax, ymax) = approximate_bounding_box(obj, scene)
            bounding_boxes[i, j, 0] = xmin
            bounding_boxes[i, j, 1] = ymin
            bounding_boxes[i, j, 2] = xmax
            bounding_boxes[i, j, 3] = ymax

            # 5b) class label
            # Build the label string "size color material shape"
            obj_name = f"{obj['size']} {obj['color']} {obj['material']} {obj['shape']}"
            if obj_name in label_to_id:
                class_labels[i, j] = label_to_id[obj_name]
            else:
                # Fallback if we missed something
                class_labels[i, j] = 0

    # 6) Write to HDF5
    # We'll store 'bounding_boxes', 'class_labels', 'image_index', and
    # also store 'image_filenames' in a variable-length string dataset.
    with h5py.File(H5_OUTPUT_PATH, "w") as f:
        f.create_dataset("bounding_boxes", data=bounding_boxes)
        f.create_dataset("class_labels", data=class_labels)
        f.create_dataset("image_index", data=image_index)

        # For variable-length strings, we need special dtype
        dt = h5py.special_dtype(vlen=bytes)
        dset_filenames = f.create_dataset("image_filename", (num_scenes,), dtype=dt)
        dset_filenames[...] = image_filenames

    print(f"Saved bounding boxes to HDF5: {H5_OUTPUT_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()
