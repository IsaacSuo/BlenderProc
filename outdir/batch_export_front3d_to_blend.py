#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml


def load_config():
    config_path = Path(__file__).with_name("render_profile.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample 3D-FRONT scenes and export each one as a .blend file."
    )
    parser.add_argument("--count", type=int, default=20, help="Number of random scenes to export.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. 0 means system randomness.")
    parser.add_argument("--front-json-dir", default=None, help="Override front_json_dir from config.")
    parser.add_argument("--future-model-dir", default=None, help="Override future_model_dir from config.")
    parser.add_argument("--front-texture-dir", default=None, help="Override front_texture_dir from config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store exported .blend files. Default: <paths.output_dir>/blend_exports",
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=1.0,
        help="Uniform scene scale forwarded to export_front3d_to_blend.py.",
    )
    parser.add_argument(
        "--scale-pivot",
        choices=("floor_center", "bbox_center", "origin"),
        default="floor_center",
        help="Scale pivot forwarded to export_front3d_to_blend.py.",
    )
    parser.add_argument(
        "--lamp-light-strength",
        type=float,
        default=7.0,
        help="Built-in lamp emission strength forwarded to export_front3d_to_blend.py.",
    )
    parser.add_argument(
        "--ceiling-light-strength",
        type=float,
        default=0.8,
        help="Built-in ceiling emission strength forwarded to export_front3d_to_blend.py.",
    )
    parser.add_argument(
        "--skip-pack",
        action="store_true",
        help="Forward --skip-pack to export_front3d_to_blend.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .blend files. By default existing exports are skipped.",
    )
    parser.add_argument(
        "--blenderproc-bin",
        default="blenderproc",
        help="BlenderProc executable to use.",
    )
    return parser.parse_args()


def resolve_paths(args, config):
    paths = config.get("paths", {})
    front_json_dir = args.front_json_dir or paths.get("front_json_dir")
    future_model_dir = args.future_model_dir or paths.get("future_model_dir")
    front_texture_dir = args.front_texture_dir or paths.get("front_texture_dir")
    output_dir = args.output_dir
    if not output_dir:
        base_output_dir = paths.get("output_dir") or "./output"
        output_dir = os.path.join(base_output_dir, "blend_exports")

    resolved = {
        "front_json_dir": os.path.abspath(front_json_dir) if front_json_dir else None,
        "future_model_dir": os.path.abspath(future_model_dir) if future_model_dir else None,
        "front_texture_dir": os.path.abspath(front_texture_dir) if front_texture_dir else None,
        "output_dir": os.path.abspath(output_dir),
    }
    return resolved


def validate_paths(paths):
    if not paths["front_json_dir"] or not os.path.isdir(paths["front_json_dir"]):
        raise FileNotFoundError(f"front_json_dir not found: {paths['front_json_dir']}")
    if not paths["future_model_dir"] or not os.path.isdir(paths["future_model_dir"]):
        raise FileNotFoundError(f"future_model_dir not found: {paths['future_model_dir']}")
    if not paths["front_texture_dir"] or not os.path.isdir(paths["front_texture_dir"]):
        raise FileNotFoundError(f"front_texture_dir not found: {paths['front_texture_dir']}")
    os.makedirs(paths["output_dir"], exist_ok=True)


def choose_json_files(front_json_dir, count, seed):
    json_files = sorted(file_name for file_name in os.listdir(front_json_dir) if file_name.endswith(".json"))
    if not json_files:
        raise RuntimeError(f"No .json files found in {front_json_dir}")

    if count <= 0:
        raise ValueError("--count must be > 0")

    actual_count = min(count, len(json_files))
    rng = random.Random(None if seed == 0 else seed)
    return rng.sample(json_files, actual_count)


def build_export_command(args, paths, front_json_name, output_blend):
    export_script = Path(__file__).with_name("export_front3d_to_blend.py")
    front_json_path = os.path.join(paths["front_json_dir"], front_json_name)

    cmd = [
        args.blenderproc_bin,
        "run",
        str(export_script),
        front_json_path,
        paths["future_model_dir"],
        paths["front_texture_dir"],
        output_blend,
        "--lamp-light-strength",
        str(args.lamp_light_strength),
        "--ceiling-light-strength",
        str(args.ceiling_light_strength),
        "--scene-scale",
        str(args.scene_scale),
        "--scale-pivot",
        args.scale_pivot,
    ]
    if args.skip_pack:
        cmd.append("--skip-pack")
    return cmd


def main():
    args = parse_args()
    config = load_config()
    paths = resolve_paths(args, config)
    validate_paths(paths)

    selected_json_files = choose_json_files(paths["front_json_dir"], args.count, args.seed)
    manifest = {
        "count_requested": args.count,
        "count_selected": len(selected_json_files),
        "seed": args.seed,
        "front_json_dir": paths["front_json_dir"],
        "future_model_dir": paths["future_model_dir"],
        "front_texture_dir": paths["front_texture_dir"],
        "output_dir": paths["output_dir"],
        "scene_scale": args.scene_scale,
        "scale_pivot": args.scale_pivot,
        "results": [],
    }

    print(f"Selected {len(selected_json_files)} scenes from {paths['front_json_dir']}")

    for index, front_json_name in enumerate(selected_json_files, start=1):
        scene_stem = Path(front_json_name).stem
        output_blend = os.path.join(paths["output_dir"], f"{scene_stem}.blend")

        if os.path.exists(output_blend) and not args.overwrite:
            print(f"[{index}/{len(selected_json_files)}] Skip existing: {output_blend}")
            manifest["results"].append({
                "front_json": front_json_name,
                "output_blend": output_blend,
                "status": "skipped_existing",
            })
            continue

        cmd = build_export_command(args, paths, front_json_name, output_blend)
        print(f"[{index}/{len(selected_json_files)}] Exporting {front_json_name} -> {output_blend}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            manifest["results"].append({
                "front_json": front_json_name,
                "output_blend": output_blend,
                "status": "error",
                "returncode": exc.returncode,
            })
            print(f"  failed: returncode={exc.returncode}", file=sys.stderr)
            continue

        manifest["results"].append({
            "front_json": front_json_name,
            "output_blend": output_blend,
            "status": "ok",
        })

    manifest_path = os.path.join(paths["output_dir"], "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    ok_count = sum(1 for item in manifest["results"] if item["status"] == "ok")
    skip_count = sum(1 for item in manifest["results"] if item["status"] == "skipped_existing")
    error_count = sum(1 for item in manifest["results"] if item["status"] == "error")
    print(f"Done. ok={ok_count}, skipped={skip_count}, error={error_count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
