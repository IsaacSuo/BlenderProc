import blenderproc as bproc  # must be first import for blenderproc run
# Pre-scan script: find Top-K placement positions per 3D-FRONT scene.
# Usage:
#   blenderproc run outdir/prescan_placements.py --batch 0
#   blenderproc run outdir/prescan_placements.py --batch 1 --batch-size 500
#   blenderproc run outdir/prescan_placements.py --batch 0 --batch-size 3  # quick test
import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HERE = Path(__file__).resolve().parent
render_profile = _load_module("batch_render_profile", _HERE / "batch_render_profile.py")
_place_mod = _load_module("front3d_place_and_render", _HERE / "front3d_place_and_render.py")

SUPPORT_CATEGORY_IDS = _place_mod.SUPPORT_CATEGORY_IDS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-scan 3D-FRONT scenes: find top-K placement positions per scene."
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch index (0-based)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Scenes per batch")
    parser.add_argument("--output-dir", default="./prescan_output/", help="CSV output directory")
    parser.add_argument("--top-k", type=int, default=10, help="Top positions to keep per scene")
    parser.add_argument("--placement-candidates", type=int, default=None,
                        help="Candidates per support surface (default: from LOGIC_CONFIG)")
    parser.add_argument("--target-max-size", type=float, default=0.40,
                        help="Assumed max object size for probe height")
    parser.add_argument(
        "--scene-names",
        nargs="+",
        default=None,
        help="Optional list of scene json basenames to scan directly, bypassing batch slicing.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reused utilities (from precompute_sphere_radius.py)
# ---------------------------------------------------------------------------

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


def project_point_to_surface(surface_obj, world_point):
    """Project a point vertically down onto the sliced support surface."""
    local2world = np.array(surface_obj.get_local2world_mat())
    world2local = np.linalg.inv(local2world)

    point = np.array([float(world_point[0]), float(world_point[1]), float(world_point[2]), 1.0], dtype=float)
    origin_local = world2local @ point
    direction_local = world2local @ np.array([0.0, 0.0, -1.0, 0.0], dtype=float)

    hit, location, _, _ = surface_obj.blender_obj.ray_cast(
        Vector(origin_local[:3]),
        Vector(direction_local[:3]).normalized(),
        distance=1e6,
    )
    if not hit:
        return None

    hit_local = np.array([float(location[0]), float(location[1]), float(location[2]), 1.0], dtype=float)
    hit_world = local2world @ hit_local
    return np.array(hit_world[:3], dtype=float)


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------

def scan_all_placements(room_objs, obj_half_height, probe_directions, logic, top_k):
    """Scan all whitelist support surfaces, sample candidates, return top_k by sphere_radius."""
    n_candidates = int(logic.get("placement_candidates", 20))
    safety_margin = float(logic.get("sphere_safety_margin", 0.15))
    min_radius = float(logic.get("min_sphere_radius", 0.3))
    max_radius = float(logic.get("max_sphere_radius", 3.0))

    # Build BVH tree once for the entire scene
    bvh_objs = [o for o in room_objs if is_valid_mesh_object(o)]
    if not bvh_objs:
        return []
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(bvh_objs)

    all_results = []

    for obj in room_objs:
        try:
            cat_id = obj.get_cp("category_id")
        except Exception:
            continue
        if cat_id not in SUPPORT_CATEGORY_IDS:
            continue
        if not is_valid_mesh_object(obj):
            continue

        support_name = obj.get_name()

        try:
            surface_obj = bproc.object.slice_faces_with_normals(obj)
        except Exception:
            continue
        if surface_obj is None:
            continue

        for _ in range(n_candidates):
            try:
                sampled_loc = bproc.sampler.upper_region(
                    objects_to_sample_on=[surface_obj],
                    min_height=0.2, max_height=0.8, use_ray_trace_check=True,
                )
            except Exception:
                continue

            surface_hit = project_point_to_surface(surface_obj, sampled_loc)
            if surface_hit is None:
                continue

            probe_center = (
                float(surface_hit[0]),
                float(surface_hit[1]),
                float(surface_hit[2] + obj_half_height),
            )
            clearance = evaluate_clearance_at_position(probe_center, bvh_tree, probe_directions)
            sphere_radius = max(min(clearance - safety_margin, max_radius), 0.0)

            all_results.append({
                "support_name": support_name,
                "category_id": cat_id,
                "sphere_radius": round(sphere_radius, 4),
                "clearance": round(clearance, 4),
                "pos_x": round(float(probe_center[0]), 4),
                "pos_y": round(float(probe_center[1]), 4),
                "pos_z": round(float(probe_center[2]), 4),
                "surface_z": round(float(surface_hit[2]), 4),
                "viable": sphere_radius >= min_radius,
            })

        # Restore the original object by joining the sliced surface back
        surface_obj.join_with_other_objects([obj])

    # Sort by sphere_radius descending, take top_k
    all_results.sort(key=lambda r: r["sphere_radius"], reverse=True)
    return all_results[:top_k]


def process_one_scene(json_path, future_model_dir, front_texture_dir,
                      mapping, probe_directions, logic, obj_half_height, top_k):
    """Load one scene, scan placements, return list of top_k result dicts."""
    front_json = os.path.basename(json_path)
    try:
        room_objs = bproc.loader.load_front3d(
            json_path=json_path,
            future_model_path=future_model_dir,
            front_3D_texture_path=front_texture_dir,
            label_mapping=mapping,
        )
    except Exception as e:
        return [{
            "front_json": front_json,
            "rank": -1,
            "support_name": "",
            "category_id": "",
            "sphere_radius": "",
            "clearance": "",
            "pos_x": "", "pos_y": "", "pos_z": "",
            "surface_z": "",
            "viable": False,
            "error": str(e)[:120],
        }]

    results = scan_all_placements(room_objs, obj_half_height, probe_directions, logic, top_k)

    if not results:
        return [{
            "front_json": front_json,
            "rank": -1,
            "support_name": "",
            "category_id": "",
            "sphere_radius": "",
            "clearance": "",
            "pos_x": "", "pos_y": "", "pos_z": "",
            "surface_z": "",
            "viable": False,
            "error": "no_viable_placement",
        }]

    rows = []
    for rank, r in enumerate(results):
        rows.append({
            "front_json": front_json,
            "rank": rank,
            "support_name": r["support_name"],
            "category_id": r["category_id"],
            "sphere_radius": r["sphere_radius"],
            "clearance": r["clearance"],
            "pos_x": r["pos_x"],
            "pos_y": r["pos_y"],
            "pos_z": r["pos_z"],
            "surface_z": r["surface_z"],
            "viable": r["viable"],
            "error": "",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    paths = render_profile.PATHS_CONFIG
    logic = dict(render_profile.LOGIC_CONFIG)

    # Override placement_candidates from CLI if specified
    if args.placement_candidates is not None:
        logic["placement_candidates"] = args.placement_candidates

    front_json_dir = paths.get("front_json_dir")
    future_model_dir = paths.get("future_model_dir")
    front_texture_dir = paths.get("front_texture_dir")

    if not front_json_dir or not os.path.isdir(front_json_dir):
        raise ValueError(f"front_json_dir not found: {front_json_dir}")

    # Enumerate and sort scene files by hash name
    json_files = sorted([f for f in os.listdir(front_json_dir) if f.endswith(".json")])
    total_scenes = len(json_files)

    if args.scene_names:
        scene_set = set(args.scene_names)
        batch_files = [jf for jf in json_files if jf in scene_set]
        missing = sorted(scene_set - set(batch_files))
        if missing:
            raise FileNotFoundError(f"Scene names not found: {missing[:5]}")
        csv_name = "prescan_custom.csv"
        print(f"Total scenes: {total_scenes}")
        print(f"Custom scene subset: {len(batch_files)} scenes")
    else:
        if args.batch is None:
            raise ValueError("--batch is required unless --scene-names is provided")
        start = args.batch * args.batch_size
        end = min(start + args.batch_size, total_scenes)
        if start >= total_scenes:
            print(f"Batch {args.batch} is out of range (total {total_scenes} scenes, "
                  f"batch-size {args.batch_size}). Nothing to do.")
            return
        batch_files = json_files[start:end]
        csv_name = f"prescan_batch_{args.batch:04d}.csv"
        print(f"Total scenes: {total_scenes}")
        print(f"Batch {args.batch}: scenes [{start}, {end}) — {len(batch_files)} scenes")

    # Init Blender
    bproc.init()
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    n_probes = int(logic.get("probe_directions", 60))
    probe_directions = generate_probe_directions(n_probes)
    obj_half_height = args.target_max_size / 2.0

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, csv_name)
    fieldnames = [
        "front_json", "rank", "support_name", "category_id",
        "sphere_radius", "clearance",
        "pos_x", "pos_y", "pos_z", "surface_z",
        "viable", "error",
    ]

    all_rows = []
    errors = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        f.flush()

        for i, jf in enumerate(batch_files):
            json_path = os.path.join(front_json_dir, jf)
            rows = process_one_scene(
                json_path, future_model_dir, front_texture_dir,
                mapping, probe_directions, logic, obj_half_height, args.top_k,
            )
            all_rows.extend(rows)
            for row in rows:
                writer.writerow(row)
            f.flush()

            # Progress log
            if rows and rows[0].get("error"):
                errors += 1
                print(f"[{i+1}/{len(batch_files)}] {jf} -> ERROR: {rows[0]['error']}")
            else:
                best_r = rows[0]["sphere_radius"] if rows else "?"
                n_viable = sum(1 for r in rows if r.get("viable"))
                print(f"[{i+1}/{len(batch_files)}] {jf} -> top sphere_radius={best_r}, "
                      f"{n_viable}/{len(rows)} viable")

            blender_clean_up()

    print(f"\nDone. Processed {len(batch_files)} scenes, {errors} errors.")
    print(f"CSV: {csv_path} ({len(all_rows)} rows)")

    viable_rows = [r for r in all_rows if r.get("viable") and r.get("rank", -1) >= 0]
    print(f"Total viable placements: {len(viable_rows)}")
    if viable_rows:
        radii = [float(r["sphere_radius"]) for r in viable_rows]
        print(f"  max={max(radii):.4f}  min={min(radii):.4f}  "
              f"median={sorted(radii)[len(radii)//2]:.4f}")


if __name__ == "__main__":
    main()
