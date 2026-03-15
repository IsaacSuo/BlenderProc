import blenderproc as bproc
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from mathutils import Vector


def load_render_profile():
    module_path = Path(__file__).with_name("batch_render_profile.py")
    spec = importlib.util.spec_from_file_location("batch_render_profile", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load render profile module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_profile = load_render_profile()


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute sphere_radius for a single 3D-FRONT scene.")
    parser.add_argument("front_json", help="Path to a 3D-FRONT room json file.")
    parser.add_argument("future_model_dir", help="Path to 3D-FUTURE-model directory.")
    parser.add_argument("front_texture_dir", help="Path to 3D-FRONT-texture directory.")
    parser.add_argument("object_path", help="Path to the custom object file.")
    parser.add_argument("--support-keywords", nargs="+", default=["bed", "table", "desk"])
    parser.add_argument("--target-max-size", type=float, default=0.40)
    return parser.parse_args()


def find_support_candidates(room_objs, keywords):
    lowered = [k.lower() for k in keywords]
    candidates = []
    for obj in room_objs:
        name = obj.get_name().lower()
        matched_priority = None
        for priority, keyword in enumerate(lowered):
            if keyword in name:
                matched_priority = priority
                break
        if matched_priority is None:
            continue
        bbox = obj.get_bound_box()
        extent = np.max(bbox, axis=0) - np.min(bbox, axis=0)
        area = float(extent[0] * extent[1])
        candidates.append((matched_priority, -area, obj))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [obj for _, _, obj in candidates]


def is_valid_mesh_object(obj):
    try:
        bl = getattr(obj, "blender_obj", None)
        return bl is not None and bl.type == "MESH" and bl.data is not None
    except ReferenceError:
        return False


def generate_probe_directions(n):
    return render_profile.generate_fibonacci_points(
        n_samples=n, radius=1.0, center_loc=Vector((0, 0, 0)), hemisphere=False,
    )


def evaluate_clearance_at_position(position, bvh_tree, probe_directions):
    min_dist = float('inf')
    origin = Vector(position)
    for d in probe_directions:
        _, _, _, dist = bvh_tree.ray_cast(origin, Vector(d).normalized())
        if dist is not None and dist < min_dist:
            min_dist = dist
    return min_dist


def precompute_for_support(support_obj, custom_obj, room_objs, probe_directions, logic):
    n_candidates = int(logic.get("placement_candidates", 20))
    safety_margin = float(logic.get("sphere_safety_margin", 0.15))
    min_radius = float(logic.get("min_sphere_radius", 0.3))
    max_radius = float(logic.get("max_sphere_radius", 3.0))

    support_name = support_obj.get_name()
    surface_obj = bproc.object.slice_faces_with_normals(support_obj)
    if surface_obj is None:
        return None

    bvh_objs = [o for o in room_objs if o != custom_obj and is_valid_mesh_object(o)]
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(bvh_objs)

    obj_bbox = custom_obj.get_bound_box()
    obj_half_height = float((np.max(obj_bbox, axis=0) - np.min(obj_bbox, axis=0))[2]) / 2.0
    surface_center_z = float(np.mean(surface_obj.get_bound_box(), axis=0)[2])

    best_clearance = 0.0
    for _ in range(n_candidates):
        try:
            sampled_loc = bproc.sampler.upper_region(
                objects_to_sample_on=[surface_obj],
                min_height=0.2, max_height=0.8, use_ray_trace_check=False,
            )
        except Exception:
            continue
        probe_center = Vector((sampled_loc[0], sampled_loc[1], surface_center_z + obj_half_height))
        clearance = evaluate_clearance_at_position(probe_center, bvh_tree, probe_directions)
        if clearance > best_clearance:
            best_clearance = clearance

    surface_obj.join_with_other_objects([support_obj])

    sphere_radius = max(min(best_clearance - safety_margin, max_radius), 0.0)
    return {
        "support_name": support_name,
        "best_clearance": round(best_clearance, 4),
        "sphere_radius": round(sphere_radius, 4),
        "viable": sphere_radius >= min_radius,
    }


def main():
    args = parse_args()
    bproc.init()

    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    room_objs = bproc.loader.load_front3d(
        json_path=args.front_json,
        future_model_path=args.future_model_dir,
        front_3D_texture_path=args.front_texture_dir,
        label_mapping=mapping,
    )

    support_candidates = find_support_candidates(room_objs, args.support_keywords)

    custom_obj = bproc.loader.load_obj(args.object_path)
    if not custom_obj:
        print(json.dumps({"front_json": args.front_json, "error": "failed_to_load_object"}), flush=True)
        return
    from blenderproc.python.types.MeshObjectUtility import MeshObject
    import bmesh, bpy
    valid = [o for o in custom_obj if getattr(o, "blender_obj", None) and o.blender_obj.type == "MESH"]
    if not valid:
        print(json.dumps({"front_json": args.front_json, "error": "no_valid_mesh"}), flush=True)
        return
    obj = valid[0]
    bbox = obj.get_bound_box()
    extent = np.max(bbox, axis=0) - np.min(bbox, axis=0)
    largest = float(np.max(extent))
    if largest > 0:
        scale = args.target_max_size / largest
        obj.set_scale(np.array(obj.blender_obj.scale) * scale)
        obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

    logic = render_profile.LOGIC_CONFIG
    n_probes = int(logic.get("probe_directions", 60))
    probe_directions = generate_probe_directions(n_probes)

    best_result = None
    for support_obj in support_candidates:
        if not is_valid_mesh_object(support_obj):
            continue
        result = precompute_for_support(support_obj, obj, room_objs, probe_directions, logic)
        if result is None:
            continue
        if best_result is None or result["sphere_radius"] > best_result["sphere_radius"]:
            best_result = result

    output = {
        "front_json": os.path.basename(args.front_json),
        "num_supports": len(support_candidates),
    }
    if best_result:
        output.update(best_result)
    else:
        output["error"] = "no_viable_support"

    print("PRECOMPUTE_RESULT:" + json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
