import blenderproc as bproc

import argparse
import json
import os
import random
from pathlib import Path

import bpy
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch render 3D-FRONT scenes with BlenderProc."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(Path(__file__).with_name("render.yaml")),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--scene",
        dest="scene_name",
        default=None,
        help="Render a single scene JSON by file name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override the maximum number of scenes to render.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def resolve_path(path_value):
    if not path_value:
        return None
    return str(Path(path_value).expanduser())


def require_path(path_value, key_name, expect_dir=False, expect_file=False):
    path_value = resolve_path(path_value)
    if not path_value:
        raise ValueError(f"Missing config path: {key_name}")
    if expect_dir and not os.path.isdir(path_value):
        raise FileNotFoundError(f"{key_name} is not a directory: {path_value}")
    if expect_file and not os.path.isfile(path_value):
        raise FileNotFoundError(f"{key_name} is not a file: {path_value}")
    return os.path.abspath(path_value)


def list_scene_jsons(front_json_dir, config, args):
    available = sorted(
        file_name for file_name in os.listdir(front_json_dir) if file_name.endswith(".json")
    )
    if not available:
        raise FileNotFoundError(f"No .json files found in {front_json_dir}")

    if args.scene_name:
        if args.scene_name not in available:
            raise FileNotFoundError(f"Scene not found in {front_json_dir}: {args.scene_name}")
        return [args.scene_name]

    selection = config.get("selection", {})
    scene_names = selection.get("scene_names") or []
    if scene_names:
        missing = [scene_name for scene_name in scene_names if scene_name not in available]
        if missing:
            raise FileNotFoundError(f"Configured scenes not found: {', '.join(missing)}")
        available = [scene_name for scene_name in available if scene_name in set(scene_names)]

    seed = int(selection.get("seed", 0) or 0)
    shuffle = bool(selection.get("shuffle", False))
    if shuffle:
        rng = random.Random(seed if seed != 0 else None)
        rng.shuffle(available)

    limit = args.limit if args.limit is not None else selection.get("max_scenes")
    if limit is not None:
        limit = int(limit)
        if limit < 0:
            raise ValueError("selection.max_scenes/--limit must be >= 0")
        available = available[:limit]

    return available


def configure_scene(config):
    render_cfg = config.get("render", {})
    logic_cfg = config.get("logic", {})

    device_mode = str(render_cfg.get("device", "GPU")).upper()
    use_only_cpu = device_mode == "CPU"
    gpu_backend = render_cfg.get("gpu_backend")
    gpu_indices = render_cfg.get("gpu_indices")
    bproc.renderer.set_render_devices(
        use_only_cpu=use_only_cpu,
        desired_gpu_device_type=gpu_backend,
        desired_gpu_ids=gpu_indices,
    )
    if not use_only_cpu:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        for device in preferences.devices:
            if device.type == "CPU":
                device.use = False

    samples_max = int(render_cfg.get("samples_max", 128))
    noise_threshold = float(render_cfg.get("noise_threshold", 0.03))
    use_denoising = bool(render_cfg.get("use_denoising", True))
    denoiser = render_cfg.get("denoiser", "OPTIX" if not use_only_cpu else "INTEL")
    bproc.renderer.set_max_amount_of_samples(samples_max)
    bproc.renderer.set_noise_threshold(noise_threshold)
    bproc.renderer.set_denoiser(denoiser if use_denoising else None)

    res_x = int(render_cfg.get("res_x", 1600))
    res_y = int(render_cfg.get("res_y", 1200))
    lens = float(logic_cfg.get("lens", 50.0))
    bproc.camera.set_intrinsics_from_blender_params(
        lens=lens,
        image_width=res_x,
        image_height=res_y,
    )

    light_paths = render_cfg.get("light_paths", {})
    bproc.renderer.set_light_bounces(
        diffuse_bounces=int(light_paths.get("diffuse_bounces", 4)),
        glossy_bounces=int(light_paths.get("glossy_bounces", 4)),
        max_bounces=int(light_paths.get("max_bounces", 8)),
        transmission_bounces=int(light_paths.get("transmission", 8)),
        transparent_max_bounces=int(light_paths.get("transparent_max", 8)),
    )

    outputs_cfg = render_cfg.get("outputs", {})
    if bool(outputs_cfg.get("depth", False)):
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
    if bool(outputs_cfg.get("normals", True)):
        bproc.renderer.enable_normals_output()
    if bool(outputs_cfg.get("segmentation", True)):
        map_by = outputs_cfg.get("segmentation_map_by", ["category_id"])
        bproc.renderer.enable_segmentation_output(map_by=map_by)


def load_front3d_scene(scene_json_path, paths_cfg):
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    return bproc.loader.load_front3d(
        json_path=scene_json_path,
        future_model_path=paths_cfg["future_model_dir"],
        front_3D_texture_path=paths_cfg["front_texture_dir"],
        label_mapping=mapping,
    )


def find_special_object_ids(loaded_objects, names):
    lowered_names = [name.lower() for name in names]
    category_ids = []
    for obj in loaded_objects:
        obj_name = obj.get_name().lower()
        if any(name in obj_name for name in lowered_names):
            try:
                category_ids.append(obj.get_cp("category_id"))
            except Exception:
                continue
    return category_ids


def sample_camera_poses(loaded_objects, config):
    logic_cfg = config.get("logic", {})
    point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)
    mesh_objects = [obj for obj in loaded_objects if isinstance(obj, bproc.types.MeshObject)]
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(mesh_objects)

    special_names = logic_cfg.get("special_object_names", ["chair", "sofa", "table", "bed"])
    special_objects = find_special_object_ids(loaded_objects, special_names)

    poses_target = int(logic_cfg.get("num_views", 10))
    max_tries = int(logic_cfg.get("max_tries", 10000))
    height_min = float(logic_cfg.get("height_min", 1.4))
    height_max = float(logic_cfg.get("height_max", 1.8))
    pitch_min = float(logic_cfg.get("pitch_min", 1.2217))
    pitch_max = float(logic_cfg.get("pitch_max", 1.338))
    coverage_score_min = float(logic_cfg.get("coverage_score_min", 0.8))
    proximity_checks = {
        "min": float(logic_cfg.get("proximity_min", 1.0)),
        "avg": {
            "min": float(logic_cfg.get("proximity_avg_min", 2.5)),
            "max": float(logic_cfg.get("proximity_avg_max", 3.5)),
        },
        "no_background": bool(logic_cfg.get("proximity_no_background", True)),
    }

    poses = 0
    tries = 0
    while tries < max_tries and poses < poses_target:
        height = np.random.uniform(height_min, height_max)
        location = point_sampler.sample(height)
        rotation = np.random.uniform([pitch_min, 0.0, 0.0], [pitch_max, 0.0, np.pi * 2.0])
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

        coverage_ok = True
        if special_objects:
            coverage = bproc.camera.scene_coverage_score(
                cam2world_matrix,
                special_objects,
                special_objects_weight=10.0,
            )
            coverage_ok = coverage > coverage_score_min

        if coverage_ok and bproc.camera.perform_obstacle_in_view_check(
            cam2world_matrix, proximity_checks, bvh_tree
        ):
            bproc.camera.add_camera_pose(cam2world_matrix)
            poses += 1
        tries += 1

    return poses


def write_scene_metadata(output_dir, scene_json_path, camera_count):
    metadata = {
        "scene_json": os.path.abspath(scene_json_path),
        "camera_count": int(camera_count),
    }
    with open(os.path.join(output_dir, "scene_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def render_one_scene(scene_json_path, scene_name, config, paths_cfg, output_root, scene_index):
    if scene_index == 0:
        bproc.init()
    else:
        bproc.clean_up(clean_up_camera=True)

    configure_scene(config)
    loaded_objects = load_front3d_scene(scene_json_path, paths_cfg)
    camera_count = sample_camera_poses(loaded_objects, config)
    if camera_count <= 0:
        raise RuntimeError("No camera poses were accepted for this scene.")

    data = bproc.renderer.render()
    output_dir = os.path.join(output_root, Path(scene_name).stem)
    os.makedirs(output_dir, exist_ok=True)
    bproc.writer.write_hdf5(output_dir, data)
    write_scene_metadata(output_dir, scene_json_path, camera_count)

    return {
        "scene_name": scene_name,
        "scene_json": scene_json_path,
        "output_dir": output_dir,
        "camera_count": camera_count,
        "status": "ok",
    }


def main():
    args = parse_args()
    config_path = require_path(args.config, "config", expect_file=True)
    config = load_config(config_path)

    paths_cfg = config.get("paths", {})
    front_json_dir = require_path(paths_cfg.get("front_json_dir"), "paths.front_json_dir", expect_dir=True)
    future_model_dir = require_path(paths_cfg.get("future_model_dir"), "paths.future_model_dir", expect_dir=True)
    front_texture_dir = require_path(paths_cfg.get("front_texture_dir"), "paths.front_texture_dir", expect_dir=True)
    output_root = os.path.abspath(resolve_path(paths_cfg.get("output_dir") or "./output/3df"))
    os.makedirs(output_root, exist_ok=True)

    runtime_paths = {
        "front_json_dir": front_json_dir,
        "future_model_dir": future_model_dir,
        "front_texture_dir": front_texture_dir,
    }

    scene_names = list_scene_jsons(front_json_dir, config, args)
    base_seed = int(config.get("selection", {}).get("seed", 0) or 0)

    summary = {
        "config_path": config_path,
        "front_json_dir": front_json_dir,
        "future_model_dir": future_model_dir,
        "front_texture_dir": front_texture_dir,
        "output_dir": output_root,
        "scene_count": len(scene_names),
        "results": [],
    }

    for scene_index, scene_name in enumerate(scene_names):
        scene_json_path = os.path.join(front_json_dir, scene_name)
        scene_seed = base_seed + scene_index if base_seed != 0 else None
        if scene_seed is not None:
            random.seed(scene_seed)
            np.random.seed(scene_seed)

        print(f"[{scene_index + 1}/{len(scene_names)}] Rendering {scene_name}")
        try:
            result = render_one_scene(
                scene_json_path=scene_json_path,
                scene_name=scene_name,
                config=config,
                paths_cfg=runtime_paths,
                output_root=output_root,
                scene_index=scene_index,
            )
        except Exception as exc:
            result = {
                "scene_name": scene_name,
                "scene_json": scene_json_path,
                "status": "error",
                "error": str(exc),
            }
            print(f"  failed: {exc}")
        else:
            print(f"  ok: {result['camera_count']} views -> {result['output_dir']}")

        summary["results"].append(result)

    summary_path = os.path.join(output_root, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
