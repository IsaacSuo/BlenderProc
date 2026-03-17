import blenderproc as bproc
import argparse
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from blenderproc.python.types.MeshObjectUtility import MeshObject


def load_render_profile():
    module_path = Path(__file__).with_name("batch_render_profile.py")
    spec = importlib.util.spec_from_file_location("batch_render_profile", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load render profile module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_profile = load_render_profile()


# 3D-FRONT category_id 白名单：适合放置物体的平面
# 按优先级分三梯队，梯队内按面积降序排列
SUPPORT_CATEGORY_IDS = {
    # --- 第一梯队：桌/床/柜顶面（大面积水平面）---
    36,   # table
    68,   # dining table
    3,    # tea table
    81,   # desk
    80,   # dressing table
    72,   # corner/side table
    12,   # round end table
    17,   # bed
    67,   # double bed
    70,   # single bed
    20,   # kids bed
    87,   # bunk bed
    88,   # bed frame
    76,   # nightstand
    56,   # tv stand
    28,   # sideboard / side cabinet / console
    74,   # shelf
    69,   # cabinet/shelf/desk
    48,   # cabinet
    6,    # children cabinet
    47,   # kitchen cabinet
    55,   # drawer chest / corner cabinet
    91,   # bookcase / jewelry armoire
    22,   # storage unit
    23,   # media unit
    84,   # wardrobe
    65,   # wine cooler
    # --- 第二梯队：沙发/椅类（可放但面积较小或曲面）---
    46,   # sofa
    18,   # two-seat sofa
    89,   # three-seat / multi-person sofa
    96,   # l-shaped sofa
    9,    # chaise longue sofa
    10,   # lazy sofa
    16,   # armchair
    50,   # chair
    14,   # dining chair
    78,   # lounge chair / book-chair / computer chair
    71,   # classic chinese chair
    83,   # dressing chair
    94,   # barstool
    35,   # pier/stool
    25,   # footstool / sofastool / bed end stool / stool
    66,   # outdoor furniture
    # --- 第三梯队：地面 ---
    51,   # floor
}

# category_id → 优先级（数值越小越优先）
_SUPPORT_PRIORITY = {}
for _prio, _cid in enumerate(SUPPORT_CATEGORY_IDS):
    _SUPPORT_PRIORITY[_cid] = _prio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Place a single-mesh custom object into a 3D-FRONT room and render it from sampled cameras."
    )
    parser.add_argument("front_json", nargs="?", help="Path to a single 3D-FRONT room json file.")
    parser.add_argument("future_model_dir", nargs="?", help="Path to the 3D-FUTURE-model directory.")
    parser.add_argument("front_texture_dir", nargs="?", help="Path to the 3D-FRONT-texture directory.")
    parser.add_argument("object_path", nargs="?", help="Path to the custom object file.")
    parser.add_argument("output_dir", nargs="?", help="Path to the output directory.")
    parser.add_argument(
        "--target-max-size",
        type=float,
        default=0.40,
        help="Scale the custom object so its largest bbox dimension matches this size in meters.",
    )
    return parser.parse_args()


def resolve_runtime_paths(args, rng):
    config_paths = render_profile.PATHS_CONFIG
    front_json = args.front_json or config_paths.get("front_json")
    if not front_json:
        front_json_dir = config_paths.get("front_json_dir")
        if not front_json_dir or not os.path.isdir(front_json_dir):
            raise ValueError("Must provide front_json (file) or front_json_dir (directory) in config")
        json_files = sorted([f for f in os.listdir(front_json_dir) if f.endswith(".json")])
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {front_json_dir}")
        front_json = os.path.join(front_json_dir, rng.choice(json_files))
    resolved = {
        "front_json": front_json,
        "future_model_dir": args.future_model_dir or config_paths.get("future_model_dir"),
        "front_texture_dir": args.front_texture_dir or config_paths.get("front_texture_dir"),
        "object_path": args.object_path or config_paths.get("object_path"),
        "output_dir": args.output_dir or config_paths.get("output_dir"),
    }
    missing = [key for key, value in resolved.items() if not value]
    if missing:
        raise ValueError(f"Missing required path config: {', '.join(missing)}")
    return resolved


def validate_paths(paths):
    for path in [paths["front_json"], paths["future_model_dir"], paths["front_texture_dir"], paths["object_path"]]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)


def find_support_candidates(room_objs):
    candidates = []
    for obj in room_objs:
        try:
            cat_id = obj.get_cp("category_id")
        except Exception:
            continue
        if cat_id not in SUPPORT_CATEGORY_IDS:
            continue
        priority = _SUPPORT_PRIORITY[cat_id]
        bbox = obj.get_bound_box()
        extent = np.max(bbox, axis=0) - np.min(bbox, axis=0)
        horizontal_area = float(extent[0] * extent[1])
        candidates.append((priority, -horizontal_area, obj))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [obj for _, _, obj in candidates]


def merge_mesh_objects(mesh_objects):
    valid_mesh_objects = [
        mesh_object for mesh_object in mesh_objects
        if getattr(mesh_object, "blender_obj", None) is not None
        and getattr(mesh_object.blender_obj, "type", None) == "MESH"
        and getattr(mesh_object.blender_obj, "data", None) is not None
    ]

    if not valid_mesh_objects:
        raise RuntimeError("No valid mesh geometry found after import.")

    if len(valid_mesh_objects) == 1:
        return valid_mesh_objects[0]

    merged_bm = bmesh.new()
    merged_materials = []
    merged_material_index_map = {}

    for mesh_object in valid_mesh_objects:
        blender_obj = mesh_object.blender_obj
        source_mesh = blender_obj.data

        local_material_map = {}
        for src_index, material in enumerate(source_mesh.materials):
            key = material.as_pointer() if material is not None else ("__none__", src_index)
            if key not in merged_material_index_map:
                merged_material_index_map[key] = len(merged_materials)
                merged_materials.append(material)
            local_material_map[src_index] = merged_material_index_map[key]

        vert_offset = len(merged_bm.verts)
        face_offset = len(merged_bm.faces)
        merged_bm.from_mesh(source_mesh)

        new_verts = list(merged_bm.verts)[vert_offset:]
        bmesh.ops.transform(merged_bm, verts=new_verts, matrix=blender_obj.matrix_world)

        new_faces = list(merged_bm.faces)[face_offset:]
        for face in new_faces:
            face.material_index = local_material_map.get(face.material_index, 0)

    merged_mesh_data = bpy.data.meshes.new(name="Merged_Custom_Mesh")
    merged_bm.to_mesh(merged_mesh_data)
    merged_bm.free()

    merged_blender_obj = bpy.data.objects.new("Merged_Custom_Object", merged_mesh_data)
    bpy.context.scene.collection.objects.link(merged_blender_obj)
    for material in merged_materials:
        merged_mesh_data.materials.append(material)

    for mesh_object in valid_mesh_objects:
        mesh_object.delete()

    return MeshObject(merged_blender_obj)


def load_custom_object(object_path):
    loaded = bproc.loader.load_obj(object_path)
    if not loaded:
        raise RuntimeError(f"Failed to load custom object: {object_path}")

    obj = merge_mesh_objects(loaded)
    obj.set_origin(np.mean(obj.get_bound_box(), axis=0), mode="POINT")
    obj.set_cp("category_id", 999)
    obj.set_cp("is_custom_object", True)
    return obj


def scale_object_to_target_size(obj, target_max_size):
    bbox = obj.get_bound_box()
    extent = np.max(bbox, axis=0) - np.min(bbox, axis=0)
    largest_dim = float(np.max(extent))
    if largest_dim <= 0:
        raise RuntimeError(f"Object has invalid bounding box: {obj.get_name()}")
    scale_factor = target_max_size / largest_dim
    obj.set_scale(np.array(obj.blender_obj.scale) * scale_factor)
    obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
    return scale_factor


def place_object_on_surface(custom_obj, support_obj):
    support_name = support_obj.get_name()
    surface_obj = bproc.object.slice_faces_with_normals(support_obj)
    if surface_obj is None:
        return {"ok": False, "reason": "no_upward_surface_extracted", "support_name": support_name}

    def sample_pose(obj):
        obj.set_location(
            bproc.sampler.upper_region(
                objects_to_sample_on=[surface_obj],
                min_height=0.2,
                max_height=0.8,
                use_ray_trace_check=False,
            )
        )
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    placed_objects = bproc.object.sample_poses_on_surface(
        objects_to_sample=[custom_obj],
        surface=surface_obj,
        sample_pose_func=sample_pose,
        min_distance=0.0,
        max_distance=10.0,
        check_all_bb_corners_over_surface=False,
    )
    if not placed_objects:
        surface_obj.join_with_other_objects([support_obj])
        return {"ok": False, "reason": "surface_sampler_failed", "support_name": support_name}

    if render_profile.LOGIC_CONFIG.get("use_physics", False):
        custom_obj.enable_rigidbody(True, collision_shape="CONVEX_HULL")
        surface_obj.enable_rigidbody(False)
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=2,
            max_simulation_time=4,
            check_object_interval=1,
        )

    surface_height_z = float(np.mean(surface_obj.get_bound_box(), axis=0)[2])
    object_bottom_z = float(np.min(custom_obj.get_bound_box(local_coords=False), axis=0)[2])
    surface_name = surface_obj.get_name()
    surface_obj.join_with_other_objects([support_obj])

    if abs(object_bottom_z - surface_height_z) > 0.05:
        return {"ok": False, "reason": "object_bottom_not_aligned_with_surface", "support_name": support_name}

    return {
        "ok": True,
        "support_name": support_name,
        "surface_name": surface_name,
    }


def generate_probe_directions(n_directions):
    return render_profile.generate_fibonacci_points(
        n_samples=n_directions,
        radius=1.0,
        center_loc=Vector((0, 0, 0)),
        hemisphere=False,
    )


def evaluate_clearance_at_position(position, bvh_tree, probe_directions):
    min_dist = float('inf')
    origin = Vector(position)
    for direction in probe_directions:
        _, _, _, dist = bvh_tree.ray_cast(origin, Vector(direction).normalized())
        if dist is not None and dist < min_dist:
            min_dist = dist
    return min_dist


def place_object_on_surface_space_aware(custom_obj, support_obj, room_objs):
    logic = render_profile.LOGIC_CONFIG
    n_candidates = int(logic.get("placement_candidates", 20))
    n_probes = int(logic.get("probe_directions", 60))
    safety_margin = float(logic.get("sphere_safety_margin", 0.15))
    min_radius = float(logic.get("min_sphere_radius", 0.5))
    max_radius = float(logic.get("max_sphere_radius", 3.0))

    support_name = support_obj.get_name()
    surface_obj = bproc.object.slice_faces_with_normals(support_obj)
    if surface_obj is None:
        return {"ok": False, "reason": "no_upward_surface_extracted", "support_name": support_name}

    probe_directions = generate_probe_directions(n_probes)

    bvh_objs = [obj for obj in room_objs if obj != custom_obj and _is_valid_mesh_object(obj)]
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(bvh_objs)

    obj_bbox = custom_obj.get_bound_box()
    obj_extent = np.max(obj_bbox, axis=0) - np.min(obj_bbox, axis=0)
    obj_half_height = float(obj_extent[2]) / 2.0

    surface_bbox = surface_obj.get_bound_box()
    surface_center_z = float(np.mean(surface_bbox, axis=0)[2])

    candidates = []
    for _ in range(n_candidates):
        sampled_loc = bproc.sampler.upper_region(
            objects_to_sample_on=[surface_obj],
            min_height=0.2,
            max_height=0.8,
            use_ray_trace_check=False,
        )
        probe_center = Vector((sampled_loc[0], sampled_loc[1], surface_center_z + obj_half_height))
        clearance = evaluate_clearance_at_position(probe_center, bvh_tree, probe_directions)
        candidates.append((clearance, sampled_loc))

    candidates.sort(key=lambda c: c[0], reverse=True)

    for clearance, sampled_loc in candidates:
        if clearance - safety_margin < min_radius:
            continue

        def sample_pose(obj):
            obj.set_location(sampled_loc)
            obj.set_rotation_euler(bproc.sampler.uniformSO3())

        placed_objects = bproc.object.sample_poses_on_surface(
            objects_to_sample=[custom_obj],
            surface=surface_obj,
            sample_pose_func=sample_pose,
            min_distance=0.0,
            max_distance=10.0,
            check_all_bb_corners_over_surface=False,
        )
        if not placed_objects:
            continue

        if render_profile.LOGIC_CONFIG.get("use_physics", False):
            custom_obj.enable_rigidbody(True, collision_shape="CONVEX_HULL")
            surface_obj.enable_rigidbody(False)
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=2,
                max_simulation_time=4,
                check_object_interval=1,
            )

        surface_height_z = float(np.mean(surface_obj.get_bound_box(), axis=0)[2])
        object_bottom_z = float(np.min(custom_obj.get_bound_box(local_coords=False), axis=0)[2])
        surface_name = surface_obj.get_name()

        if abs(object_bottom_z - surface_height_z) > 0.05:
            continue

        actual_center = np.mean(custom_obj.get_bound_box(local_coords=False), axis=0)
        actual_clearance = evaluate_clearance_at_position(actual_center, bvh_tree, probe_directions)
        sphere_radius = max(min(actual_clearance - safety_margin, max_radius), 0.0)
        if sphere_radius < min_radius:
            continue

        surface_obj.join_with_other_objects([support_obj])
        return {
            "ok": True,
            "support_name": support_name,
            "surface_name": surface_name,
            "sphere_radius": sphere_radius,
        }

    surface_obj.join_with_other_objects([support_obj])
    return {"ok": False, "reason": "no_candidate_with_sufficient_clearance", "support_name": support_name}


def _is_valid_mesh_object(obj):
    try:
        bl = getattr(obj, "blender_obj", None)
        return bl is not None and bl.type == "MESH" and bl.data is not None
    except ReferenceError:
        return False


def apply_batch_render_material_strategy(obj, object_path):
    material_params = render_profile.sample_material_params_for_object(object_path)
    obj.blender_obj["_material_params_json"] = json.dumps(material_params, ensure_ascii=False)
    render_profile.apply_material(obj.blender_obj)


def add_batch_render_camera_poses(anchor_obj, target_obj, room_objs, sphere_radius=None):
    render_profile.setup_render_settings()

    scene = bpy.context.scene
    cam_obj = render_profile.create_smart_camera(anchor_obj.blender_obj)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=render_profile.LOGIC_CONFIG["lens"],
        image_width=render_profile.RENDER_CONFIG["res_x"],
        image_height=render_profile.RENDER_CONFIG["res_y"],
    )

    target_bbox = target_obj.get_bound_box()
    target_location = np.mean(target_bbox, axis=0)
    target_extent = np.max(target_bbox, axis=0) - np.min(target_bbox, axis=0)
    target_size = float(np.max(target_extent))
    focus_jitter = max(target_size * render_profile.LOGIC_CONFIG.get("focus_jitter_factor", 0.15), 0.02)

    if sphere_radius is not None:
        camera_distance = sphere_radius
        candidate_count = render_profile.LOGIC_CONFIG["num_views"]
    else:
        target_radius = render_profile.LOGIC_CONFIG["target_diameter"] / 2.0
        margin = render_profile.LOGIC_CONFIG["margin"]
        fov_h = cam_obj.data.angle
        aspect_ratio = scene.render.resolution_x / scene.render.resolution_y
        fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect_ratio)
        camera_distance = (target_radius * margin) / math.sin(min(fov_h, fov_v) / 2)
        candidate_multiplier = max(int(render_profile.LOGIC_CONFIG.get("candidate_view_multiplier", 6)), 1)
        candidate_count = max(
            render_profile.LOGIC_CONFIG["num_views"] * candidate_multiplier,
            render_profile.LOGIC_CONFIG["num_views"],
        )

    camera_positions_local = render_profile.generate_fibonacci_points(
        n_samples=candidate_count,
        radius=camera_distance,
        center_loc=Vector((0, 0, 0)),
        hemisphere=render_profile.LOGIC_CONFIG["use_hemisphere"],
    )

    if sphere_radius is None:
        proximity_checks = {
            "min": camera_distance * render_profile.LOGIC_CONFIG.get("proximity_min_factor", 0.6),
            "avg": {
                "min": camera_distance * render_profile.LOGIC_CONFIG.get("proximity_avg_min_factor", 1.0),
                "max": camera_distance * render_profile.LOGIC_CONFIG.get("proximity_avg_max_factor", 0.9),
            },
            "no_background": True,
        }
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(
            [obj for obj in room_objs if _is_valid_mesh_object(obj)]
        )

    accepted_views = 0
    for pos_local in camera_positions_local:
        cam_obj.location = pos_local + anchor_obj.blender_obj.location
        bpy.context.view_layer.update()
        focus_target = target_location + np.random.uniform(-focus_jitter, focus_jitter, size=3)
        toward_direction = focus_target - np.array(cam_obj.location)
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            toward_direction,
            inplane_rot=np.random.uniform(
                render_profile.LOGIC_CONFIG.get("inplane_rot_min_rad", -0.5),
                render_profile.LOGIC_CONFIG.get("inplane_rot_max_rad", 0.5),
            ),
        )
        cam2world_matrix = bproc.math.build_transformation_mat(np.array(cam_obj.location), rotation_matrix)

        if sphere_radius is None:
            if not bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
                continue
        if target_obj not in bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=20):
            continue

        bproc.camera.add_camera_pose(np.array(cam2world_matrix))
        accepted_views += 1
        if accepted_views >= render_profile.LOGIC_CONFIG["num_views"]:
            break

    if accepted_views == 0:
        raise RuntimeError("No valid camera pose found for the placed object.")

    return accepted_views


def add_binary_mask_from_category_id(data, target_category_id=999):
    category_segmaps = data.get("category_id_segmaps")
    if category_segmaps is None:
        raise KeyError("category_id_segmaps is missing; cannot derive binary mask.")
    data["binary_masks"] = [(segmap == target_category_id).astype(np.uint8) * 255 for segmap in category_segmaps]
    return data


def write_metadata(
    output_dir, front_json, object_path, support_name, surface_name,
    scale_factor, camera_count, placement_mode="traditional", sphere_radius=None,
):
    metadata = {
        "front_json": os.path.abspath(front_json),
        "object_path": os.path.abspath(object_path),
        "support_object_name": support_name,
        "surface_object_name": surface_name,
        "scale_factor": scale_factor,
        "camera_count": camera_count,
        "placement_mode": placement_mode,
    }
    if sphere_radius is not None:
        metadata["sphere_radius"] = sphere_radius
    with open(os.path.join(output_dir, "placement_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main():
    args = parse_args()
    master_seed = int(render_profile.LOGIC_CONFIG.get("master_seed", 0))
    rng = random.Random(master_seed if master_seed != 0 else None)
    paths = resolve_runtime_paths(args, rng)
    validate_paths(paths)
    os.makedirs(paths["output_dir"], exist_ok=True)

    bproc.init()

    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    bproc.renderer.set_light_bounces(
        diffuse_bounces=200,
        glossy_bounces=200,
        max_bounces=200,
        transmission_bounces=200,
        transparent_max_bounces=200,
    )

    room_objs = bproc.loader.load_front3d(
        json_path=paths["front_json"],
        future_model_path=paths["future_model_dir"],
        front_3D_texture_path=paths["front_texture_dir"],
        label_mapping=mapping,
        lamp_light_strength=0,  # disable built-in emission; lamp_lighting module handles it
    )

    # Apply real lights for lamp objects
    import lamp_lighting
    lamp_lighting.apply_lighting_to_scene(room_objs, paths["front_json"])

    support_candidates = find_support_candidates(room_objs)
    if not support_candidates:
        raise RuntimeError("No support object found matching SUPPORT_CATEGORY_IDS whitelist.")

    custom_obj = load_custom_object(paths["object_path"])
    scale_factor = scale_object_to_target_size(custom_obj, args.target_max_size)

    placement_info = None
    sphere_radius = None
    placement_mode = "traditional"

    if render_profile.LOGIC_CONFIG.get("space_aware_placement", False):
        best_info = None
        best_radius = -1.0
        for candidate_obj in support_candidates:
            if not _is_valid_mesh_object(candidate_obj):
                continue
            info = place_object_on_surface_space_aware(custom_obj, candidate_obj, room_objs)
            if info["ok"] and info.get("sphere_radius", 0) > best_radius:
                best_info = info
                best_radius = info["sphere_radius"]
        if best_info is not None:
            placement_info = best_info
            sphere_radius = best_radius
            placement_mode = "space_aware"

    if placement_info is None:
        for candidate_obj in support_candidates:
            if not _is_valid_mesh_object(candidate_obj):
                continue
            info = place_object_on_surface(custom_obj, candidate_obj)
            if info["ok"]:
                placement_info = info
                sphere_radius = None
                placement_mode = "traditional_fallback" if render_profile.LOGIC_CONFIG.get("space_aware_placement", False) else "traditional"
                break

    if placement_info is None:
        custom_obj.delete()
        raise RuntimeError("Failed to place the custom object on any support object.")

    if sphere_radius is not None:
        surface_z = float(np.min(custom_obj.get_bound_box(local_coords=False), axis=0)[2])
        target_diameter = float(render_profile.LOGIC_CONFIG["target_diameter"])
        margin = float(render_profile.LOGIC_CONFIG["margin"])
        cam_data = bpy.data.cameras.get("UltraCam") or bpy.data.cameras.new("UltraCam")
        cam_data.lens = render_profile.LOGIC_CONFIG["lens"]
        fov_h = cam_data.angle
        aspect_ratio = render_profile.RENDER_CONFIG["res_x"] / render_profile.RENDER_CONFIG["res_y"]
        fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect_ratio)
        ideal_diameter = 2 * sphere_radius * math.sin(min(fov_h, fov_v) / 2) / margin
        current_bbox = custom_obj.get_bound_box()
        current_size = float(np.max(np.max(current_bbox, axis=0) - np.min(current_bbox, axis=0)))
        if current_size > 0:
            rescale = ideal_diameter / current_size
            custom_obj.set_scale(np.array(custom_obj.blender_obj.scale) * rescale)
            custom_obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
            scale_factor *= rescale
            new_bottom_z = float(np.min(custom_obj.get_bound_box(local_coords=False), axis=0)[2])
            loc = custom_obj.get_location()
            loc[2] += surface_z - new_bottom_z
            custom_obj.set_location(loc)

    apply_batch_render_material_strategy(custom_obj, paths["object_path"])

    anchor = bproc.object.create_primitive("CUBE")
    anchor.set_name("ANCHOR")
    anchor.hide(True)
    anchor.set_location(np.mean(custom_obj.get_bound_box(), axis=0))

    camera_count = add_batch_render_camera_poses(anchor, custom_obj, room_objs + [custom_obj], sphere_radius=sphere_radius)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

    data = bproc.renderer.render()
    data = add_binary_mask_from_category_id(data, target_category_id=999)
    bproc.writer.write_hdf5(paths["output_dir"], data)

    write_metadata(
        output_dir=paths["output_dir"],
        front_json=paths["front_json"],
        object_path=paths["object_path"],
        support_name=placement_info["support_name"],
        surface_name=placement_info["surface_name"],
        scale_factor=scale_factor,
        camera_count=camera_count,
        placement_mode=placement_mode,
        sphere_radius=sphere_radius,
    )


if __name__ == "__main__":
    main()
