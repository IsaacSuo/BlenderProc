import blenderproc as bproc
import argparse
import importlib.util
import json
import math
import os
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
        "--support-keywords",
        nargs="+",
        default=["bed", "table", "desk"],
        help="Ordered object name keywords used to rank support objects.",
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
        if matched_priority is None:
            continue
        bbox = obj.get_bound_box()
        extent = np.max(bbox, axis=0) - np.min(bbox, axis=0)
        horizontal_area = float(extent[0] * extent[1])
        candidates.append((matched_priority, -horizontal_area, obj))
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


def _build_bvh_tree_excluding(room_objs, exclude_obj):
    bm = bmesh.new()
    for obj in room_objs:
        if obj == exclude_obj:
            continue
        bl_obj = getattr(obj, "blender_obj", None)
        if bl_obj is None or bl_obj.type != "MESH" or bl_obj.data is None:
            continue
        offset = len(bm.verts)
        bm.from_mesh(bl_obj.data)
        new_verts = list(bm.verts)[offset:]
        bmesh.ops.transform(bm, verts=new_verts, matrix=bl_obj.matrix_world)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree


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
    bvh_tree = _build_bvh_tree_excluding(room_objs, custom_obj)

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
        sphere_radius = max(min(clearance - safety_margin, max_radius), 0.0)
        if sphere_radius < min_radius:
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
        raise RuntimeError(f"No support object found. Keywords tried: {', '.join(args.support_keywords)}")

    support_obj = support_candidates[0]
    support_name = support_obj.get_name()

    custom_obj = load_custom_object(paths["object_path"])
    scale_factor = scale_object_to_target_size(custom_obj, args.target_max_size)

    if render_profile.LOGIC_CONFIG.get("space_aware_placement", False):
        placement_info = place_object_on_surface_space_aware(custom_obj, support_obj, room_objs)
        sphere_radius = placement_info.get("sphere_radius") if placement_info["ok"] else None
        placement_mode = "space_aware"
    else:
        placement_info = place_object_on_surface(custom_obj, support_obj)
        sphere_radius = None
        placement_mode = "traditional"

    if not placement_info["ok"]:
        custom_obj.delete()
        raise RuntimeError(
            f"Failed to place the custom object on selected support object: "
            f"{support_name} ({placement_info['reason']})"
        )

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
