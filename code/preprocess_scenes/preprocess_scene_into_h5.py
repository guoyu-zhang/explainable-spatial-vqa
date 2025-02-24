#!/usr/bin/env python3

import json
import h5py
import numpy as np

# -------------------------------------------------------------------------
# 1. Build a unified vocabulary
# -------------------------------------------------------------------------
def build_global_vocab(shapes, colors, materials, sizes):
    """
    Creates a single, global vocab so each (category=value) is unique.

    Example entries:
      "shape=cube",
      "shape=sphere",
      "color=gray",
      "material=metal",
      "size=large", etc.

    Returns:
      - global_vocab: dict mapping string -> integer ID (starting from 1)
      - idx_to_str:   list or dict to map back from int -> string
    """

    # Step 1: Collect all possible (category=value) strings in one set
    attr_strings = set()
    for s in shapes:
        attr_strings.add(f"shape={s}")
    for c in colors:
        attr_strings.add(f"color={c}")
    for m in materials:
        attr_strings.add(f"material={m}")
    for z in sizes:
        attr_strings.add(f"size={z}")

    # Step 2: Sort and assign integer IDs
    sorted_attrs = sorted(list(attr_strings))
    global_vocab = {val: i for i, val in enumerate(sorted_attrs, start=1)}

    return global_vocab

# -------------------------------------------------------------------------
# Script Configuration
# -------------------------------------------------------------------------
JSON_SCENES_PATH = "../data/CLEVR_v1.0/scenes/CLEVR_train_scenes.json"
H5_OUTPUT_PATH   = "../h5_files/train_scenes.h5"
VOCAB_JSON_PATH  = "CLEVR_train_scenes_vocab.json"

def main():
    # -------------------------------------------------------------------------
    # 2. Load the CLEVR scenes JSON
    # -------------------------------------------------------------------------
    with open(JSON_SCENES_PATH, "r") as f:
        scene_data = json.load(f)
    
    scenes = scene_data["scenes"]  # A list of scene dicts
    num_scenes = len(scenes)
    print(f"Loaded {num_scenes} scenes from {JSON_SCENES_PATH}")

    # -------------------------------------------------------------------------
    # 3. Collect attribute values (shapes, colors, materials, sizes)
    # -------------------------------------------------------------------------
    shapes = set()
    colors = set()
    materials = set()
    sizes = set()

    for scene in scenes:
        for obj in scene["objects"]:
            shapes.add(obj["shape"])
            colors.add(obj["color"])
            materials.add(obj["material"])
            sizes.add(obj["size"])

    # -------------------------------------------------------------------------
    # 4. Build a single unified (category=value) vocabulary
    # -------------------------------------------------------------------------
    global_vocab = build_global_vocab(shapes, colors, materials, sizes)
    print(f"Total unique (category=value) entries: {len(global_vocab)}")

    # -------------------------------------------------------------------------
    # 5. Figure out max number of objects in any scene
    # -------------------------------------------------------------------------
    max_objects = max(len(scene["objects"]) for scene in scenes)
    print(f"Max number of objects in any scene: {max_objects}")

    # -------------------------------------------------------------------------
    # 6. Create arrays to store data
    # -------------------------------------------------------------------------
    # We'll store: image_index, attributes, coords_3d, coords_pixel.
    #
    # attributes[i, j, :] = [shape_code, color_code, material_code, size_code]
    #
    # shape of attributes = (num_scenes, max_objects, 4)
    #   - 4 is because each object has 4 categorical attributes in this design.
    #
    # coords_3d           = (num_scenes, max_objects, 3)
    # coords_pixel        = (num_scenes, max_objects, 3)
    #
    # image_index         = (num_scenes, )
    # -------------------------------------------------------------------------
    image_index_arr  = np.zeros((num_scenes,), dtype=np.int32)
    attributes_arr   = np.zeros((num_scenes, max_objects, 4), dtype=np.int32)
    coords_3d_arr    = np.zeros((num_scenes, max_objects, 3), dtype=np.float32)
    coords_pixel_arr = np.zeros((num_scenes, max_objects, 3), dtype=np.float32)

    # -------------------------------------------------------------------------
    # 7. Fill the arrays
    # -------------------------------------------------------------------------
    for i, scene in enumerate(scenes):
        image_index_arr[i] = scene["image_index"]
        objects = scene["objects"]

        for j, obj in enumerate(objects):
            # Build the 4 codes
            shape_str    = f"shape={obj['shape']}"
            color_str    = f"color={obj['color']}"
            material_str = f"material={obj['material']}"
            size_str     = f"size={obj['size']}"

            shape_code    = global_vocab[shape_str]
            color_code    = global_vocab[color_str]
            material_code = global_vocab[material_str]
            size_code     = global_vocab[size_str]

            attributes_arr[i, j, 0] = shape_code
            attributes_arr[i, j, 1] = color_code
            attributes_arr[i, j, 2] = material_code
            attributes_arr[i, j, 3] = size_code

            coords_3d_arr[i, j]    = obj["3d_coords"]
            coords_pixel_arr[i, j] = obj["pixel_coords"]

    # -------------------------------------------------------------------------
    # 8. Write to HDF5
    # -------------------------------------------------------------------------
    with h5py.File(H5_OUTPUT_PATH, "w") as f:
        f.create_dataset("image_index",   data=image_index_arr)
        f.create_dataset("attributes",    data=attributes_arr)
        f.create_dataset("coords_3d",     data=coords_3d_arr)
        f.create_dataset("coords_pixel",  data=coords_pixel_arr)

    print(f"HDF5 file saved to: {H5_OUTPUT_PATH}")

    # -------------------------------------------------------------------------
    # 9. Save the unified vocabulary to JSON
    # -------------------------------------------------------------------------
    with open(VOCAB_JSON_PATH, "w") as vf:
        json.dump(global_vocab, vf, indent=2)

    print(f"Vocabulary JSON saved to: {VOCAB_JSON_PATH}")

if __name__ == "__main__":
    main()
