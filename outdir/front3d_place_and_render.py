import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import blenderproc as bproc
import numpy as np
import bpy
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Place a custom GLB object into a 3D-FRONT room and render it from sampled cameras."
    )
    parser.add_argument("front_json", help="Path to a single 3D-FRONT room json file.")
    parser.add_argument("future_model_dir", help="Path to the 3D-FUTURE-model directory.")
    parser.add_argument("front_texture_dir", help="Path to the 3D-FRONT-texture directory.")
    parser.add_argument("object_path", help="Path to the custom object file (.glb/.gltf/.obj/.ply/.fbx...).")
    parser.add_argument("output_dir", help="Path to the output directory.")
    parser.add_argument(
        "--support-keywords",
        nargs="+",
        default=["bed", "table", "desk"],
        help="Ordered object name keywords used to find support surfaces in the room.",
    )
    parser.add_argument(
        "--target-max-size",
        type=float,
        default=0.45,
        help="Scale the custom object so its largest bbox dimension matches this size in meters.",
    )
    return parser.parse_args()


def validate_paths(args):
    for path in [args.front_json, args.future_model_dir, args.front_texture_dir, args.object_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)


def load_batch_render_module():
    batch_render_path = Path(__file__).with_name("batch_render.py")
    spec = importlib.util.spec_from_file_location("front3d_batch_render_strategy", batch_render_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load batch_render.py from {batch_render_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_support_candidates(room_objs, keywords):
    lowered = [keyword.lower() for keyword in keywords]
    candidates = []
    for obj in room_objs:
        name = obj.get_name().lower()
        matched_priority = None
        for priority, keyword in enumerate(lowered):
            if keyword in name:
                matched_priority = priority
                break
        if matched_priority is not None:
            bbox = obj.get_bound_box()
            extent = np.max(bbox, axis=0) - np.min(bbox, axis=0)
            horizontal_area = float(extent[0] * extent[1])
            candidates.append((matched_priority, -horizontal_area, obj))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [obj for _, _, obj in candidates]


def load_custom_object(object_path):
    loaded = bproc.loader.load_obj(object_path)
    if not loaded:
        raise RuntimeError(f"Failed to load object: {object_path}")

    root = loaded[0]
    if len(loaded) > 1:
        root.join_with_other_objects(loaded[1:])

    root.set_origin(np.mean(root.get_bound_box(), axis=0), mode="POINT")
    root.set_cp("category_id", 999)
    root.set_cp("is_custom_object", True)
    return root


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


def place_object_on_surface(custom_obj, surface_obj):
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
        return None

    placed = placed_objects[0]
    placed.enable_rigidbody(True, collision_shape="CONVEX_HULL")
    surface_obj.enable_rigidbody(False)
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=2,
        max_simulation_time=4,
        check_object_interval=1,
    )
    return placed


def object_is_on_surface(obj, surface_obj, tolerance=0.05):
    obj_min_z = float(np.min(obj.get_bound_box(local_coords=False), axis=0)[2])
    surface_top_z = float(np.max(surface_obj.get_bound_box(local_coords=False), axis=0)[2])
    return abs(obj_min_z - surface_top_z) <= tolerance


def apply_batch_render_material_strategy(batch_render, obj, object_path):
    material_params = batch_render.sample_material_params_for_object(object_path)
    obj.blender_obj["_material_params_json"] = json.dumps(material_params, ensure_ascii=False)
    batch_render.apply_material(obj.blender_obj)


def add_batch_render_camera_poses(batch_render, anchor_obj):
    batch_render.setup_render_settings()

    scene = bpy.context.scene
    cam_obj = batch_render.create_smart_camera(anchor_obj.blender_obj)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=batch_render.LOGIC_CONFIG["lens"],
        image_width=batch_render.RENDER_CONFIG["res_x"],
        image_height=batch_render.RENDER_CONFIG["res_y"],
    )

    target_radius = batch_render.LOGIC_CONFIG["target_diameter"] / 2.0
    margin = batch_render.LOGIC_CONFIG["margin"]
    fov_h = cam_obj.data.angle
    aspect_ratio = scene.render.resolution_x / scene.render.resolution_y
    fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect_ratio)
    safe_distance_3d = (target_radius * margin) / math.sin(min(fov_h, fov_v) / 2)

    camera_positions_local = batch_render.generate_fibonacci_points(
        n_samples=batch_render.LOGIC_CONFIG["num_views"],
        radius=safe_distance_3d,
        center_loc=Vector((0, 0, 0)),
        hemisphere=batch_render.LOGIC_CONFIG["use_hemisphere"],
    )

    for pos_local in camera_positions_local:
        cam_obj.location = pos_local + anchor_obj.blender_obj.location
        bpy.context.view_layer.update()
        bproc.camera.add_camera_pose(np.array(cam_obj.matrix_world))

    return len(camera_positions_local)


def write_metadata(output_dir, front_json, object_path, support_name, surface_name, scale_factor, camera_count):
    metadata = {
        "front_json": os.path.abspath(front_json),
        "object_path": os.path.abspath(object_path),
        "support_object_name": support_name,
        "surface_object_name": surface_name,
        "scale_factor": scale_factor,
        "camera_count": camera_count,
    }
    with open(os.path.join(output_dir, "placement_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def add_binary_mask_from_category_id(data, target_category_id=999):
    category_segmaps = data.get("category_id_segmaps")
    if category_segmaps is None:
        raise KeyError("category_id_segmaps is missing; cannot derive binary mask.")

    binary_masks = []
    for segmap in category_segmaps:
        binary_masks.append((segmap == target_category_id).astype(np.uint8) * 255)
    data["binary_masks"] = binary_masks
    return data


def main():
    args = parse_args()
    validate_paths(args)
    batch_render = load_batch_render_module()

    os.makedirs(args.output_dir, exist_ok=True)

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
        json_path=args.front_json,
        future_model_path=args.future_model_dir,
        front_3D_texture_path=args.front_texture_dir,
        label_mapping=mapping,
    )

    support_candidates = find_support_candidates(room_objs, args.support_keywords)
    if not support_candidates:
        raise RuntimeError(
            f"No support object found. Keywords tried: {', '.join(args.support_keywords)}"
        )

    custom_obj_template = load_custom_object(args.object_path)
    scale_factor = scale_object_to_target_size(custom_obj_template, args.target_max_size)
    custom_obj_template.hide(True)

    placed_obj = None
    selected_support_name = None
    selected_surface_name = None
    updated_room_objs = list(room_objs)
    for support_obj in support_candidates:
        surface_obj = bproc.object.slice_faces_with_normals(support_obj)
        if surface_obj is None:
            continue

        candidate_obj = custom_obj_template.duplicate(linked=False)
        candidate_obj.hide(False)
        candidate_obj.set_cp("category_id", 999)
        candidate_obj.set_cp("is_custom_object", True)

        placed_candidate = place_object_on_surface(candidate_obj, surface_obj)
        if placed_candidate is None or not object_is_on_surface(placed_candidate, surface_obj):
            candidate_obj.delete()
            surface_obj.join_with_other_objects([support_obj])
            continue

        selected_support_name = support_obj.get_name()
        selected_surface_name = surface_obj.get_name()
        surface_obj.join_with_other_objects([support_obj])
        updated_room_objs = [obj for obj in updated_room_objs if obj is not support_obj]
        updated_room_objs.append(surface_obj)
        placed_obj = placed_candidate
        break

    if placed_obj is None or selected_support_name is None:
        raise RuntimeError("Failed to place the custom object on any support surface.")

    custom_obj_template.delete()

    apply_batch_render_material_strategy(batch_render, placed_obj, args.object_path)

    anchor = bproc.object.create_primitive("CUBE")
    anchor.set_name("ANCHOR")
    anchor.hide(True)
    anchor.set_location(np.mean(placed_obj.get_bound_box(), axis=0))

    camera_count = add_batch_render_camera_poses(batch_render, anchor)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
    data = bproc.renderer.render()
    data = add_binary_mask_from_category_id(data, target_category_id=999)
    bproc.writer.write_hdf5(args.output_dir, data)

    write_metadata(
        output_dir=args.output_dir,
        front_json=args.front_json,
        object_path=args.object_path,
        support_name=selected_support_name,
        surface_name=selected_surface_name,
        scale_factor=scale_factor,
        camera_count=camera_count,
    )


if __name__ == "__main__":
    main()
