import blenderproc as bproc
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import bpy
from mathutils import Matrix, Vector


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
    parser = argparse.ArgumentParser(
        description="Place a custom GLB object into a 3D-FRONT room and render it from sampled cameras."
    )
    parser.add_argument("front_json", nargs="?", help="Path to a single 3D-FRONT room json file.")
    parser.add_argument("future_model_dir", nargs="?", help="Path to the 3D-FUTURE-model directory.")
    parser.add_argument("front_texture_dir", nargs="?", help="Path to the 3D-FRONT-texture directory.")
    parser.add_argument("object_path", nargs="?", help="Path to the custom object file (.glb/.gltf/.obj/.ply/.fbx...).")
    parser.add_argument("output_dir", nargs="?", help="Path to the output directory.")
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


def resolve_runtime_paths(args):
    config_paths = render_profile.PATHS_CONFIG
    resolved = {
        "front_json": args.front_json or config_paths.get("front_json"),
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


def get_group_bbox(mesh_objects):
    bbox_points = [obj.get_bound_box() for obj in mesh_objects]
    all_points = np.concatenate(bbox_points, axis=0)
    return np.min(all_points, axis=0), np.max(all_points, axis=0)


def get_group_center(mesh_objects):
    bbox_min, bbox_max = get_group_bbox(mesh_objects)
    return (bbox_min + bbox_max) / 2.0


def get_group_extent(mesh_objects):
    bbox_min, bbox_max = get_group_bbox(mesh_objects)
    return bbox_max - bbox_min


def translate_group(mesh_objects, offset):
    offset = np.asarray(offset, dtype=float)
    for obj in mesh_objects:
        obj.set_location(obj.get_location() + offset)


def transform_group(mesh_objects, transform_matrix):
    for obj in mesh_objects:
        obj.set_local2world_mat(np.asarray(transform_matrix) @ obj.get_local2world_mat())


def set_group_custom_properties(mesh_objects):
    for obj in mesh_objects:
        obj.set_cp("category_id", 999)
        obj.set_cp("is_custom_object", True)


def hide_group(mesh_objects, hide=True):
    for obj in mesh_objects:
        obj.hide(hide)


def delete_group(mesh_objects):
    for obj in mesh_objects:
        obj.delete()


def load_custom_object(object_path):
    loaded = bproc.loader.load_obj(object_path)
    if not loaded:
        raise RuntimeError(f"Failed to load object: {object_path}")

    set_group_custom_properties(loaded)
    return loaded


def scale_object_to_target_size(mesh_objects, target_max_size):
    extent = get_group_extent(mesh_objects)
    largest_dim = float(np.max(extent))
    if largest_dim <= 0:
        raise RuntimeError("Object group has invalid bounding box.")

    scale_factor = target_max_size / largest_dim
    center = get_group_center(mesh_objects)
    scale_matrix = Matrix.Diagonal((scale_factor, scale_factor, scale_factor, 1.0))
    transform = Matrix.Translation(center) @ scale_matrix @ Matrix.Translation(-Vector(center))
    transform_group(mesh_objects, transform)
    return scale_factor


def place_object_on_surface(mesh_objects, surface_obj):
    surface_bbox = surface_obj.get_bound_box()
    surface_center = np.mean(surface_bbox, axis=0)
    surface_top_z = float(np.max(surface_bbox[:, 2]))

    group_center = get_group_center(mesh_objects)
    bbox_min, _ = get_group_bbox(mesh_objects)
    offset = np.array([
        surface_center[0] - group_center[0],
        surface_center[1] - group_center[1],
        surface_top_z - float(bbox_min[2]),
    ])
    translate_group(mesh_objects, offset)
    return mesh_objects


def object_is_on_surface(mesh_objects, surface_obj, tolerance=0.05):
    bbox_min, _ = get_group_bbox(mesh_objects)
    obj_min_z = float(bbox_min[2])
    surface_top_z = float(np.max(surface_obj.get_bound_box(local_coords=False), axis=0)[2])
    return abs(obj_min_z - surface_top_z) <= tolerance


def apply_batch_render_material_strategy(mesh_objects, object_path):
    material_params = render_profile.sample_material_params_for_object(object_path)
    for obj in mesh_objects:
        obj.blender_obj["_material_params_json"] = json.dumps(material_params, ensure_ascii=False)
        render_profile.apply_material(obj.blender_obj)


def add_batch_render_camera_poses(anchor_obj):
    render_profile.setup_render_settings()

    scene = bpy.context.scene
    cam_obj = render_profile.create_smart_camera(anchor_obj.blender_obj)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=render_profile.LOGIC_CONFIG["lens"],
        image_width=render_profile.RENDER_CONFIG["res_x"],
        image_height=render_profile.RENDER_CONFIG["res_y"],
    )

    target_radius = render_profile.LOGIC_CONFIG["target_diameter"] / 2.0
    margin = render_profile.LOGIC_CONFIG["margin"]
    fov_h = cam_obj.data.angle
    aspect_ratio = scene.render.resolution_x / scene.render.resolution_y
    fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect_ratio)
    safe_distance_3d = (target_radius * margin) / math.sin(min(fov_h, fov_v) / 2)

    camera_positions_local = render_profile.generate_fibonacci_points(
        n_samples=render_profile.LOGIC_CONFIG["num_views"],
        radius=safe_distance_3d,
        center_loc=Vector((0, 0, 0)),
        hemisphere=render_profile.LOGIC_CONFIG["use_hemisphere"],
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
    paths = resolve_runtime_paths(args)
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
    )

    support_candidates = find_support_candidates(room_objs, args.support_keywords)
    if not support_candidates:
        raise RuntimeError(
            f"No support object found. Keywords tried: {', '.join(args.support_keywords)}"
        )

    custom_obj_template = load_custom_object(paths["object_path"])
    scale_factor = scale_object_to_target_size(custom_obj_template, args.target_max_size)
    delete_group(custom_obj_template)

    placed_group = None
    selected_support_name = None
    selected_surface_name = None
    updated_room_objs = list(room_objs)
    for support_obj in support_candidates:
        surface_obj = bproc.object.slice_faces_with_normals(support_obj)
        if surface_obj is None:
            continue

        candidate_group = load_custom_object(paths["object_path"])
        scale_object_to_target_size(candidate_group, args.target_max_size)

        placed_candidate = place_object_on_surface(candidate_group, surface_obj)
        if placed_candidate is None or not object_is_on_surface(placed_candidate, surface_obj):
            delete_group(candidate_group)
            surface_obj.join_with_other_objects([support_obj])
            continue

        selected_support_name = support_obj.get_name()
        selected_surface_name = surface_obj.get_name()
        surface_obj.join_with_other_objects([support_obj])
        updated_room_objs = [obj for obj in updated_room_objs if obj is not support_obj]
        updated_room_objs.append(surface_obj)
        placed_group = placed_candidate
        break

    if placed_group is None or selected_support_name is None:
        raise RuntimeError("Failed to place the custom object on any support surface.")

    apply_batch_render_material_strategy(placed_group, paths["object_path"])

    anchor = bproc.object.create_primitive("CUBE")
    anchor.set_name("ANCHOR")
    anchor.hide(True)
    anchor.set_location(get_group_center(placed_group))

    camera_count = add_batch_render_camera_poses(anchor)

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
        support_name=selected_support_name,
        surface_name=selected_surface_name,
        scale_factor=scale_factor,
        camera_count=camera_count,
    )


if __name__ == "__main__":
    main()
