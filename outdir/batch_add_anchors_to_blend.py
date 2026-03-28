#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch add ANCHOR objects to .blend files and export anchored copies."
    )
    parser.add_argument(
        "input_path",
        help="Input .blend file or directory containing .blend files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for anchored .blend outputs. Default: <input>_anchored or sibling directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.blend",
        help="Filename glob used when input_path is a directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing anchored .blend files.",
    )
    parser.add_argument(
        "--blender-bin",
        default="blender",
        help="Blender executable to use.",
    )
    parser.add_argument(
        "--include-secondary-supports",
        action="store_true",
        help="Also place anchors on sofa/chair/stool-like supports.",
    )
    parser.add_argument(
        "--include-floor",
        action="store_true",
        help="Also place anchors on floor objects.",
    )
    parser.add_argument(
        "--include-beds",
        action="store_true",
        help="Also place anchors on bed-like supports.",
    )
    parser.add_argument(
        "--replace-existing-anchors",
        action="store_true",
        help="Remove existing ANCHOR* objects before writing the output blend.",
    )
    parser.add_argument("--anchor-display-size", type=float, default=1.0)
    parser.add_argument("--anchor-scale", type=float, default=0.5)
    parser.add_argument("--surface-angle-deg", type=float, default=15.0)
    parser.add_argument("--top-band", type=float, default=0.03)
    parser.add_argument("--z-offset", type=float, default=0.5)
    return parser.parse_args()


def resolve_blend_files(input_path, pattern):
    resolved = Path(input_path).expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".blend":
            raise ValueError(f"Input file is not a .blend: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input path not found: {resolved}")
    files = sorted(p for p in resolved.glob(pattern) if p.is_file())
    if not files:
        raise RuntimeError(f"No .blend files matched {pattern} in {resolved}")
    return files


def resolve_output_dir(input_path, output_dir):
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    resolved = Path(input_path).expanduser().resolve()
    if resolved.is_file():
        return resolved.parent / f"{resolved.stem}_anchored"
    return resolved.parent / f"{resolved.name}_anchored"


def build_command(args, input_blend, output_blend):
    anchor_script = Path(__file__).with_name("add_anchors_to_blend.py").resolve()
    cmd = [
        args.blender_bin,
        "--background",
        str(input_blend),
        "--python",
        str(anchor_script),
        "--",
        "--output-blend",
        str(output_blend),
        "--anchor-display-size",
        str(args.anchor_display_size),
        "--anchor-scale",
        str(args.anchor_scale),
        "--surface-angle-deg",
        str(args.surface_angle_deg),
        "--top-band",
        str(args.top_band),
        "--z-offset",
        str(args.z_offset),
    ]
    if args.include_secondary_supports:
        cmd.append("--include-secondary-supports")
    if args.include_floor:
        cmd.append("--include-floor")
    if args.include_beds:
        cmd.append("--include-beds")
    if args.replace_existing_anchors:
        cmd.append("--replace-existing-anchors")
    return cmd


def main():
    args = parse_args()
    blend_files = resolve_blend_files(args.input_path, args.pattern)
    output_dir = resolve_output_dir(args.input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_path": str(Path(args.input_path).expanduser().resolve()),
        "output_dir": str(output_dir),
        "blend_count": len(blend_files),
        "results": [],
    }

    for index, input_blend in enumerate(blend_files, start=1):
        output_blend = output_dir / input_blend.name
        if output_blend.exists() and not args.overwrite:
            print(f"[{index}/{len(blend_files)}] Skip existing: {output_blend}")
            manifest["results"].append({
                "input_blend": str(input_blend),
                "output_blend": str(output_blend),
                "status": "skipped_existing",
            })
            continue

        cmd = build_command(args, input_blend, output_blend)
        print(f"[{index}/{len(blend_files)}] Anchoring {input_blend.name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result = {
            "input_blend": str(input_blend),
            "output_blend": str(output_blend),
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
        }
        if proc.stdout.strip():
            result["stdout"] = proc.stdout.strip()
        if proc.stderr.strip():
            result["stderr"] = proc.stderr.strip()
        manifest["results"].append(result)

        if proc.returncode != 0:
            print(proc.stderr.strip(), file=sys.stderr)
        else:
            print(f"  wrote: {output_blend}")

    manifest_path = output_dir / "anchor_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    ok_count = sum(1 for item in manifest["results"] if item["status"] == "ok")
    skip_count = sum(1 for item in manifest["results"] if item["status"] == "skipped_existing")
    error_count = sum(1 for item in manifest["results"] if item["status"] == "error")
    print(f"Done. ok={ok_count}, skipped={skip_count}, error={error_count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
