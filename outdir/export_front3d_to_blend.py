import blenderproc as bproc

import argparse
import os
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a 3D-FRONT scene and save it as a packed .blend file for local inspection."
    )
    parser.add_argument("front_json", help="Path to a single 3D-FRONT json file.")
    parser.add_argument("future_model_dir", help="Path to the 3D-FUTURE-model directory.")
    parser.add_argument("front_texture_dir", help="Path to the 3D-FRONT-texture directory.")
    parser.add_argument("output_blend", help="Path to the output .blend file.")
    parser.add_argument(
        "--lamp-light-strength",
        type=float,
        default=7.0,
        help="Built-in lamp emission strength used by load_front3d.",
    )
    parser.add_argument(
        "--ceiling-light-strength",
        type=float,
        default=0.8,
        help="Built-in ceiling emission strength used by load_front3d.",
    )
    parser.add_argument(
        "--skip-pack",
        action="store_true",
        help="Do not pack external resources into the output .blend.",
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=1.0,
        help="Uniform scale factor applied to the exported scene before saving.",
    )
    parser.add_argument(
        "--scale-pivot",
        choices=("floor_center", "bbox_center", "origin"),
        default="floor_center",
        help="Pivot used for uniform scene scaling.",
    )
    return parser.parse_args()


def validate_paths(args):
    for path in [args.front_json, args.future_model_dir, args.front_texture_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    output_parent = os.path.dirname(os.path.abspath(args.output_blend))
    os.makedirs(output_parent, exist_ok=True)


def remove_non_scene_mesh_objects(scene_mesh_objects):
    """Delete mesh objects that were imported as temporary furniture prototypes but are not part of the final scene."""
    keep_ids = {obj.blender_obj.as_pointer() for obj in scene_mesh_objects}
    remove_objects = []
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        if obj.as_pointer() not in keep_ids:
            remove_objects.append(obj)

    for obj in remove_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)

    return len(remove_objects)


def compute_scene_pivot(scene_mesh_objects, pivot_mode):
    if not scene_mesh_objects:
        raise RuntimeError("No scene mesh objects were loaded; cannot compute scaling pivot.")

    if pivot_mode == "origin":
        return Vector((0.0, 0.0, 0.0))

    all_points = []
    for obj in scene_mesh_objects:
        all_points.append(np.asarray(obj.get_bound_box(local_coords=False), dtype=float))

    points = np.concatenate(all_points, axis=0)
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    if pivot_mode == "bbox_center":
        return Vector(((min_xyz + max_xyz) * 0.5).tolist())

    return Vector((
        float((min_xyz[0] + max_xyz[0]) * 0.5),
        float((min_xyz[1] + max_xyz[1]) * 0.5),
        float(min_xyz[2]),
    ))


def uniformly_scale_scene(scene_mesh_objects, scale, pivot):
    if scale <= 0:
        raise ValueError(f"--scene-scale must be > 0, got {scale}")
    if abs(scale - 1.0) < 1e-8:
        return 0

    transform = (
        Matrix.Translation(pivot) @
        Matrix.Scale(float(scale), 4) @
        Matrix.Translation(-pivot)
    )

    transformed = 0
    for obj in list(bpy.context.scene.objects):
        obj.matrix_world = transform @ obj.matrix_world
        transformed += 1

    bpy.context.view_layer.update()
    return transformed


def compute_compensated_emission_strength(base_strength, scene_scale):
    if scene_scale <= 0:
        raise ValueError(f"--scene-scale must be > 0, got {scene_scale}")
    return float(base_strength) * (float(scene_scale) ** 2)


def main():
    args = parse_args()
    validate_paths(args)

    bproc.init()

    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    compensated_lamp_strength = compute_compensated_emission_strength(
        args.lamp_light_strength,
        args.scene_scale,
    )
    compensated_ceiling_strength = compute_compensated_emission_strength(
        args.ceiling_light_strength,
        args.scene_scale,
    )

    scene_mesh_objects = bproc.loader.load_front3d(
        json_path=args.front_json,
        future_model_path=args.future_model_dir,
        front_3D_texture_path=args.front_texture_dir,
        label_mapping=mapping,
        ceiling_light_strength=compensated_ceiling_strength,
        lamp_light_strength=compensated_lamp_strength,
    )
    removed_count = remove_non_scene_mesh_objects(scene_mesh_objects)

    bpy.context.view_layer.update()
    scale_pivot = compute_scene_pivot(scene_mesh_objects, args.scale_pivot)
    scaled_count = uniformly_scale_scene(scene_mesh_objects, args.scene_scale, scale_pivot)

    if not args.skip_pack:
        bpy.ops.file.pack_all()

    output_path = os.path.abspath(args.output_blend)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)

    print(f"Saved blend: {output_path}")
    print(f"Source json: {os.path.abspath(args.front_json)}")
    print(f"Packed resources: {not args.skip_pack}")
    print(f"Removed temporary mesh objects: {removed_count}")
    print(f"Scene scale: {args.scene_scale}")
    print(f"Lamp emission strength: {compensated_lamp_strength}")
    print(f"Ceiling emission strength: {compensated_ceiling_strength}")
    print(f"Scale pivot mode: {args.scale_pivot}")
    print(f"Scale pivot: {tuple(scale_pivot)}")
    print(f"Transformed scene objects: {scaled_count}")


if __name__ == "__main__":
    main()
