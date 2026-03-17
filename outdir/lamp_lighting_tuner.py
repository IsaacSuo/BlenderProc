"""Lamp lighting tuner: load a single lamp model, create light, render previews.

Usage:
    blenderproc run outdir/lamp_lighting_tuner.py --jid 06bb1295-131b-471a-8e97-66251b0f79fd
    blenderproc run outdir/lamp_lighting_tuner.py --jid <JID> --energy 200 --light-type AREA --save
"""

import argparse
import importlib.util
import math
import os
from pathlib import Path

import bpy
import numpy as np
import yaml

import blenderproc as bproc
from blenderproc.python.types.LightUtility import Light

# Import sibling modules
_module_dir = Path(__file__).parent


def _import_sibling(name):
    path = _module_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lamp_lighting = _import_sibling("lamp_lighting")
render_profile = _import_sibling("batch_render_profile")


def parse_args():
    parser = argparse.ArgumentParser(description="Preview and tune lamp lighting parameters.")
    parser.add_argument("--jid", required=True, help="3D-FUTURE model jid for the lamp")
    parser.add_argument("--energy", type=float, help="Light energy override")
    parser.add_argument("--color", type=float, nargs=3, help="Light color R G B (0-1)")
    parser.add_argument("--light-type", choices=["POINT", "AREA", "SPOT", "SUN"], help="Light type override")
    parser.add_argument("--offset", type=float, nargs=3, help="Offset X Y Z from bbox center")
    parser.add_argument("--radius", type=float, help="Shadow soft size override")
    parser.add_argument("--emission-strength", type=float, help="Emissive material strength override")
    parser.add_argument("--save", action="store_true", help="Write parameters back to lamp_config.yaml")
    parser.add_argument("--output-dir", type=str, help="Preview output directory")
    return parser.parse_args()


def find_model_path(jid):
    """Locate raw_model.obj for a given jid."""
    future_model_dir = render_profile.PATHS_CONFIG.get("future_model_dir", "")
    obj_path = os.path.join(future_model_dir, jid, "raw_model.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Model not found: {obj_path}")
    return obj_path


def setup_scene():
    """Create a minimal scene: ground plane + dark environment."""
    # Dark world background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("TunerWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0

    # Ground plane: 2m x 2m, grey material
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "TunerGround"
    mat = bpy.data.materials.new("GroundGrey")
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1.0)
        principled.inputs["Roughness"].default_value = 0.8
    ground.data.materials.append(mat)


def load_lamp_model(obj_path):
    """Load lamp model and center it at origin with bottom at Z=0."""
    objs = bproc.loader.load_obj(obj_path)
    if not objs:
        raise RuntimeError(f"Failed to load model from {obj_path}")

    # Use the first object or merge if multiple
    lamp_obj = objs[0]
    if len(objs) > 1:
        # Simple approach: parent others to the first
        for extra in objs[1:]:
            extra.blender_obj.parent = lamp_obj.blender_obj

    # Center and align bottom to Z=0
    bbox = lamp_obj.get_bound_box(local_coords=False)
    center = np.mean(bbox, axis=0)
    bottom_z = np.min(bbox, axis=0)[2]
    loc = lamp_obj.get_location()
    loc[0] -= center[0]
    loc[1] -= center[1]
    loc[2] -= bottom_z
    lamp_obj.set_location(loc)
    bpy.context.view_layer.update()

    return lamp_obj


def setup_cameras(lamp_obj):
    """Place 4 cameras around the lamp: front, side, top-down, 45-degree."""
    bbox = lamp_obj.get_bound_box(local_coords=False)
    center = np.mean(bbox, axis=0)
    extent = np.max(bbox, axis=0) - np.min(bbox, axis=0)
    diag = float(np.linalg.norm(extent))
    dist = max(diag * 2.0, 1.0)

    camera_positions = [
        ("front", center + np.array([0, -dist, center[2] * 0.5])),
        ("side", center + np.array([dist, 0, center[2] * 0.5])),
        ("top", center + np.array([0, -0.01, dist])),
        ("angle45", center + np.array([dist * 0.6, -dist * 0.6, dist * 0.7])),
    ]

    bproc.camera.set_intrinsics_from_blender_params(
        lens=35.0,
        image_width=400,
        image_height=300,
    )

    for name, pos in camera_positions:
        toward = center - pos
        rot_mat = bproc.camera.rotation_from_forward_vec(toward)
        cam2world = bproc.math.build_transformation_mat(pos, rot_mat)
        bproc.camera.add_camera_pose(cam2world)


def setup_render():
    """Configure low-quality render for fast preview."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    cycles.samples = 64
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x = 400
    scene.render.resolution_y = 300

    # Try GPU
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type != "CPU"
        cycles.device = "GPU"
    except Exception:
        try:
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            for device in prefs.devices:
                device.use = device.type != "CPU"
            cycles.device = "GPU"
        except Exception:
            cycles.device = "CPU"


def apply_cli_overrides(params, args):
    """Apply CLI argument overrides to resolved params."""
    if args.energy is not None:
        params["energy"] = args.energy
    if args.color is not None:
        params["color"] = list(args.color)
    if args.light_type is not None:
        params["light_type"] = args.light_type
    if args.offset is not None:
        params["offset"] = list(args.offset)
    if args.radius is not None:
        params["radius"] = args.radius
    if args.emission_strength is not None:
        params["emission_strength"] = args.emission_strength
    return params


def save_to_config(jid, params):
    """Write current params to lamp_config.yaml models section."""
    config_path = lamp_lighting.CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if config.get("models") is None:
        config["models"] = {}

    # Only save fields that differ from resolved defaults
    defaults = dict(config.get("defaults", {}))
    save_params = {}
    for key in ["light_type", "energy", "color", "radius", "offset",
                 "keep_emission", "emission_strength", "area_size"]:
        if key in params:
            val = params[key]
            default_val = defaults.get(key)
            if val != default_val:
                save_params[key] = val

    if save_params:
        config["models"][jid] = save_params

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[tuner] Saved params for {jid} to {config_path}")
    print(f"[tuner] Saved fields: {save_params}")


def main():
    args = parse_args()
    jid = args.jid

    obj_path = find_model_path(jid)
    output_dir = args.output_dir or os.path.join(".", "lamp_previews", jid)
    os.makedirs(output_dir, exist_ok=True)

    bproc.init()

    # Scene setup
    setup_scene()
    lamp_obj = load_lamp_model(obj_path)

    # Resolve params and apply CLI overrides
    params = lamp_lighting._resolve_params(jid, "")
    params = apply_cli_overrides(params, args)

    # Create the light
    light = lamp_lighting.create_light_for_lamp(lamp_obj, params, {"jid": jid, "category": ""})

    # Cameras and render
    setup_cameras(lamp_obj)
    setup_render()

    print(f"[tuner] Model: {jid}")
    print(f"[tuner] Model path: {obj_path}")
    print(f"[tuner] Parameters: {params}")
    print(f"[tuner] Light position: {list(light.get_location())}")

    # Render
    bproc.renderer.set_light_bounces(
        diffuse_bounces=10,
        glossy_bounces=10,
        max_bounces=10,
        transmission_bounces=10,
        transparent_max_bounces=10,
    )
    data = bproc.renderer.render()

    # Save as PNG
    for i, img in enumerate(data["colors"]):
        view_names = ["front", "side", "top", "angle45"]
        view_name = view_names[i] if i < len(view_names) else f"view_{i}"
        out_path = os.path.join(output_dir, f"{view_name}.png")
        import PIL.Image
        pil_img = PIL.Image.fromarray(img)
        pil_img.save(out_path)
        print(f"[tuner] Saved: {out_path}")

    # Save config if requested
    if args.save:
        save_to_config(jid, params)

    print(f"[tuner] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
