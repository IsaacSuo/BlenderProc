import blenderproc as bproc
import argparse
import os

import bpy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a multi-mesh asset into a single mesh for clean BlenderProc placement."
    )
    parser.add_argument("input_path", help="Path to the source asset (.glb/.gltf/.obj/.fbx/.ply/...).")
    parser.add_argument("output_path", help="Path to the output asset (.glb/.gltf/.obj/.fbx/.ply).")
    parser.add_argument(
        "--origin-mode",
        default="GEOMETRY_BOUNDS",
        choices=["KEEP", "GEOMETRY_BOUNDS"],
        help="How to reset the merged object's origin.",
    )
    return parser.parse_args()


def load_mesh_objects(input_path):
    loaded = bproc.loader.load_obj(input_path)
    if not loaded:
        raise RuntimeError(f"Failed to load asset: {input_path}")
    return loaded


def merge_mesh_objects(mesh_objects):
    if len(mesh_objects) == 1:
        return mesh_objects[0]

    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    blender_objs = [obj.blender_obj for obj in mesh_objects]
    active_obj = blender_objs[0]

    for obj in blender_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active_obj

    with bpy.context.temp_override(
        active_object=active_obj,
        object=active_obj,
        selected_objects=blender_objs,
        selected_editable_objects=blender_objs,
    ):
        bpy.ops.object.join()

    return mesh_objects[0]


def reset_origin(obj, origin_mode):
    if origin_mode == "KEEP":
        return
    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = obj.blender_obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")


def export_selected_object(obj, output_path):
    output_ext = os.path.splitext(output_path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    obj.blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = obj.blender_obj

    if output_ext == ".glb":
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB", use_selection=True)
    elif output_ext == ".gltf":
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLTF_SEPARATE", use_selection=True)
    elif output_ext == ".obj":
        bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=True)
    elif output_ext == ".fbx":
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True)
    elif output_ext == ".ply":
        bpy.ops.wm.ply_export(filepath=output_path, export_selected_objects=True)
    else:
        raise RuntimeError(
            f"Unsupported output extension: {output_ext}. Use .glb, .gltf, .obj, .fbx, or .ply."
        )


def main():
    args = parse_args()
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(args.input_path)

    bproc.init()
    mesh_objects = load_mesh_objects(args.input_path)
    merged = merge_mesh_objects(mesh_objects)
    reset_origin(merged, args.origin_mode)
    export_selected_object(merged, args.output_path)

    print(
        f"Merged {len(mesh_objects)} mesh object(s) into a single asset and wrote: {args.output_path}"
    )


if __name__ == "__main__":
    main()
