import argparse
import csv
import glob
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a clean placement file from prescan CSV outputs."
    )
    parser.add_argument(
        "--input-glob",
        default="./prescan_output/prescan_batch_*.csv",
        help="Glob for prescan batch CSV files.",
    )
    parser.add_argument(
        "--output-file",
        default="./prescan_locations.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--mode",
        choices=("best-per-scene", "top-global"),
        default="best-per-scene",
        help="Whether to keep one best placement per scene or all placements.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Keep only the top N rows after sorting. 0 means keep all.",
    )
    parser.add_argument(
        "--min-clearance",
        type=float,
        default=0.0,
        help="Drop rows whose clearance is below this threshold.",
    )
    parser.add_argument(
        "--max-abs-coordinate",
        type=float,
        default=1e6,
        help="Filter out obvious bad rows with absurd coordinates.",
    )
    return parser.parse_args()


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_rows(input_glob, min_clearance, max_abs_coordinate):
    rows = []
    for path in sorted(glob.glob(input_glob)):
        with open(path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("error"):
                    continue
                if str(row.get("viable", "")).lower() != "true":
                    continue

                clearance = _safe_float(row.get("clearance"))
                sphere_radius = _safe_float(row.get("sphere_radius"))
                pos_x = _safe_float(row.get("pos_x"))
                pos_y = _safe_float(row.get("pos_y"))
                pos_z = _safe_float(row.get("pos_z"))
                surface_z = _safe_float(row.get("surface_z"))
                rank = row.get("rank")
                try:
                    rank = int(rank)
                except Exception:
                    rank = -1

                numeric_values = (clearance, sphere_radius, pos_x, pos_y, pos_z, surface_z)
                if any(v is None or not math.isfinite(v) for v in numeric_values):
                    continue
                if max(abs(pos_x), abs(pos_y), abs(pos_z), abs(surface_z)) > max_abs_coordinate:
                    continue
                if clearance < min_clearance:
                    continue

                clean = {
                    "front_json": row["front_json"],
                    "rank": rank,
                    "support_name": row["support_name"],
                    "category_id": row["category_id"],
                    "sphere_radius": sphere_radius,
                    "clearance": clearance,
                    "pos_x": pos_x,
                    "pos_y": pos_y,
                    "pos_z": pos_z,
                    "surface_z": surface_z,
                    "viable": True,
                    "source_csv": os.path.basename(path),
                }
                rows.append(clean)
    return rows


def select_rows(rows, mode):
    rows = sorted(rows, key=lambda r: (r["clearance"], r["sphere_radius"]), reverse=True)
    if mode == "top-global":
        return rows

    best_per_scene = {}
    for row in rows:
        front_json = row["front_json"]
        if front_json not in best_per_scene:
            best_per_scene[front_json] = row
    return list(best_per_scene.values())


def write_rows(rows, output_file):
    fieldnames = [
        "placement_id",
        "front_json",
        "rank",
        "support_name",
        "category_id",
        "sphere_radius",
        "clearance",
        "pos_x",
        "pos_y",
        "pos_z",
        "surface_z",
        "viable",
        "source_csv",
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({
                "placement_id": idx,
                "front_json": row["front_json"],
                "rank": row["rank"],
                "support_name": row["support_name"],
                "category_id": row["category_id"],
                "sphere_radius": f"{row['sphere_radius']:.4f}",
                "clearance": f"{row['clearance']:.4f}",
                "pos_x": f"{row['pos_x']:.4f}",
                "pos_y": f"{row['pos_y']:.4f}",
                "pos_z": f"{row['pos_z']:.4f}",
                "surface_z": f"{row['surface_z']:.4f}",
                "viable": "True",
                "source_csv": row["source_csv"],
            })


def main():
    args = parse_args()
    rows = load_rows(args.input_glob, args.min_clearance, args.max_abs_coordinate)
    if not rows:
        raise RuntimeError(f"No valid rows found for {args.input_glob}")

    selected = select_rows(rows, args.mode)
    selected.sort(key=lambda r: (r["clearance"], r["sphere_radius"]), reverse=True)
    if args.top_n > 0:
        selected = selected[:args.top_n]

    write_rows(selected, args.output_file)

    print(f"Input rows kept: {len(rows)}")
    print(f"Output rows: {len(selected)}")
    print(f"Saved placement file: {args.output_file}")
    for idx, row in enumerate(selected[:10], 1):
        print(
            f"{idx}. {row['front_json']} | {row['support_name']} | "
            f"clearance={row['clearance']:.4f} | sphere={row['sphere_radius']:.4f}"
        )


if __name__ == "__main__":
    main()
