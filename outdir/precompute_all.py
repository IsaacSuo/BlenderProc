#!/usr/bin/env python3
"""Batch precompute sphere_radius for all 3D-FRONT scenes."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config():
    config_path = Path(__file__).with_name("render_profile.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Batch precompute sphere_radius for all 3D-FRONT scenes.")
    parser.add_argument("--output-csv", default="sphere_radius_report.csv", help="Output CSV path.")
    parser.add_argument("--front-json-dir", default=None, help="Override front_json_dir from config.")
    parser.add_argument("--future-model-dir", default=None)
    parser.add_argument("--front-texture-dir", default=None)
    parser.add_argument("--object-path", default=None)
    parser.add_argument("--target-max-size", type=float, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    args = parser.parse_args()

    config = load_config()
    paths = config.get("paths", {})

    front_json_dir = args.front_json_dir or paths.get("front_json_dir")
    future_model_dir = args.future_model_dir or paths.get("future_model_dir")
    front_texture_dir = args.front_texture_dir or paths.get("front_texture_dir")
    object_path = args.object_path or paths.get("object_path")

    if not front_json_dir or not os.path.isdir(front_json_dir):
        print(f"Error: front_json_dir not found: {front_json_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted([f for f in os.listdir(front_json_dir) if f.endswith(".json")])
    print(f"Found {len(json_files)} scenes in {front_json_dir}")

    script_path = str(Path(__file__).with_name("precompute_sphere_radius.py"))
    results = []
    errors = 0

    def run_one(index_and_jf):
        i, jf = index_and_jf
        json_path = os.path.join(front_json_dir, jf)
        cmd = ["blenderproc", "run", script_path, json_path, future_model_dir, front_texture_dir, object_path]
        if args.target_max_size is not None:
            cmd += ["--target-max-size", str(args.target_max_size)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            for line in proc.stdout.splitlines():
                if line.startswith("PRECOMPUTE_RESULT:"):
                    data = json.loads(line[len("PRECOMPUTE_RESULT:"):])
                    return i, jf, data, None
            return i, jf, None, "no result"
        except subprocess.TimeoutExpired:
            return i, jf, None, "timeout"
        except Exception as e:
            return i, jf, None, str(e)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    total = len(json_files)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, (i, jf)): jf for i, jf in enumerate(json_files)}
        for future in as_completed(futures):
            i, jf, data, err = future.result()
            if data:
                results.append(data)
                r = data.get("sphere_radius", "N/A")
                print(f"[{len(results)+errors}/{total}] {jf} -> sphere_radius={r}")
            else:
                errors += 1
                print(f"[{len(results)+errors}/{total}] {jf} -> {err}")

    # Sort by sphere_radius descending
    results.sort(key=lambda r: r.get("sphere_radius", 0), reverse=True)

    # Write CSV
    fieldnames = ["front_json", "support_name", "best_clearance", "sphere_radius", "viable", "num_supports", "error"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nDone. {len(results)} scenes processed, {errors} errors.")
    print(f"CSV written to: {args.output_csv}")

    viable = [r for r in results if r.get("viable")]
    print(f"Viable scenes (sphere_radius >= min): {len(viable)} / {len(results)}")
    if viable:
        radii = [r["sphere_radius"] for r in viable]
        print(f"  max: {max(radii):.4f}  min: {min(radii):.4f}  median: {sorted(radii)[len(radii)//2]:.4f}")


if __name__ == "__main__":
    main()
