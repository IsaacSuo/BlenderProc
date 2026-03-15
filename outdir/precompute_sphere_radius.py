import blenderproc as bproc
import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import bpy
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
    parser = argparse.ArgumentParser(description="Batch precompute sphere_radius for all 3D-FRONT scenes.")
    parser.add_argument("--output-csv", default="sphere_radius_report.csv")
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


def precompute_for_support(support_obj, obj_half_height, room_objs, probe_directions, logic):
    n_candidates = int(logic.get("placement_candidates", 20))
    safety_margin = float(logic.get("sphere_safety_margin", 0.15))
    min_radius = float(logic.get("min_sphere_radius", 0.3))
    max_radius = float(logic.get("max_sphere_radius", 3.0))

    support_name = support_obj.get_name()
    surface_obj = bproc.object.slice_faces_with_normals(support_obj)
    if surface_obj is None:
        return None

    bvh_objs = [o for o in room_objs if is_valid_mesh_object(o)]
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(bvh_objs)

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


def blender_clean_up():
    """Remove all objects, meshes, materials, images from Blender to prepare for next scene."""
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for img in bpy.data.images:
        bpy.data.images.remove(img)
    for cam in bpy.data.cameras:
        bpy.data.cameras.remove(cam)
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)


def process_one_scene(json_path, args, mapping, probe_directions, logic, obj_half_height):
    """Process a single scene, return result dict or None."""
    try:
        room_objs = bproc.loader.load_front3d(
            json_path=json_path,
            future_model_path=args.future_model_dir,
            front_3D_texture_path=args.front_texture_dir,
            label_mapping=mapping,
        )
    except Exception as e:
        return {"front_json": os.path.basename(json_path), "error": str(e)[:80]}

    support_candidates = find_support_candidates(room_objs, args.support_keywords)

    best_result = None
    for support_obj in support_candidates:
        if not is_valid_mesh_object(support_obj):
            continue
        try:
            result = precompute_for_support(support_obj, obj_half_height, room_objs, probe_directions, logic)
        except Exception:
            continue
        if result is None:
            continue
        if best_result is None or result["sphere_radius"] > best_result["sphere_radius"]:
            best_result = result

    output = {
        "front_json": os.path.basename(json_path),
        "num_supports": len(support_candidates),
    }
    if best_result:
        output.update(best_result)
    else:
        output["error"] = "no_viable_support"
    return output


def main():
    args = parse_args()
    paths = render_profile.PATHS_CONFIG
    front_json_dir = paths.get("front_json_dir")
    future_model_dir = paths.get("future_model_dir")
    front_texture_dir = paths.get("front_texture_dir")
    object_path = paths.get("object_path")

    if not front_json_dir or not os.path.isdir(front_json_dir):
        raise ValueError(f"front_json_dir not found: {front_json_dir}")

    bproc.init()

    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    logic = render_profile.LOGIC_CONFIG
    n_probes = int(logic.get("probe_directions", 60))
    probe_directions = generate_probe_directions(n_probes)

    obj_half_height = args.target_max_size / 2.0

    json_files = sorted([f for f in os.listdir(front_json_dir) if f.endswith(".json")])
    total = len(json_files)
    print(f"Found {total} scenes in {front_json_dir}")

    results = []
    errors = 0

    for i, jf in enumerate(json_files):
        json_path = os.path.join(args.front_json_dir, jf)
        result = process_one_scene(json_path, args, mapping, probe_directions, logic, obj_half_height)
        results.append(result)

        r = result.get("sphere_radius", None)
        err = result.get("error", None)
        if err:
            errors += 1
            print(f"[{i+1}/{total}] {jf} -> ERROR: {err}")
        else:
            print(f"[{i+1}/{total}] {jf} -> sphere_radius={r}")

        # Clean up Blender data for next scene
        blender_clean_up()

    # Sort by sphere_radius descending
    results.sort(key=lambda r: r.get("sphere_radius", 0) or 0, reverse=True)

    # Write CSV
    fieldnames = ["front_json", "support_name", "best_clearance", "sphere_radius", "viable", "num_supports", "error"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nDone. {len(results)} scenes, {errors} errors.")
    print(f"CSV: {args.output_csv}")

    viable = [r for r in results if r.get("viable")]
    print(f"Viable: {len(viable)} / {len(results)}")
    if viable:
        radii = [r["sphere_radius"] for r in viable]
        print(f"  max={max(radii):.4f}  min={min(radii):.4f}  median={sorted(radii)[len(radii)//2]:.4f}")


if __name__ == "__main__":
    main()
