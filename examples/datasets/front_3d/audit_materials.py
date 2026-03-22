import blenderproc as bproc

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a small sample of 3D-FUTURE furniture imports used by sampled 3D-FRONT scenes."
    )
    parser.add_argument("front_json_dir", help="Directory containing 3D-FRONT scene json files.")
    parser.add_argument("future_model_dir", help="Directory containing 3D-FUTURE-model folders.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="examples/datasets/front_3d/audit_output",
        help="Directory where CSV and JSON reports will be written.",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=3,
        help="How many scene json files to sample.",
    )
    parser.add_argument(
        "--max-furniture-per-scene",
        type=int,
        default=8,
        help="How many furniture entries to sample from each selected scene.",
    )
    parser.add_argument(
        "--max-total-models",
        type=int,
        default=20,
        help="Hard cap for the total number of sampled furniture models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for reproducible sampling. 0 means system randomness.",
    )
    parser.add_argument(
        "--allow-duplicate-jids",
        action="store_true",
        help="Allow importing the same 3D-FUTURE jid more than once across sampled scenes.",
    )
    return parser.parse_args()


def validate_args(args):
    if not os.path.isdir(args.front_json_dir):
        raise FileNotFoundError(args.front_json_dir)
    if not os.path.isdir(args.future_model_dir):
        raise FileNotFoundError(args.future_model_dir)
    if args.num_scenes <= 0:
        raise ValueError("--num-scenes must be > 0")
    if args.max_furniture_per_scene <= 0:
        raise ValueError("--max-furniture-per-scene must be > 0")
    if args.max_total_models <= 0:
        raise ValueError("--max-total-models must be > 0")


def resolve_category(entry: dict) -> str:
    category = entry.get("category") or entry.get("title") or "others"
    if "/" in category:
        category = category.split("/")[0]
    return category


def list_scene_jsons(front_json_dir: str) -> List[Path]:
    return sorted(Path(front_json_dir).glob("*.json"))


def extract_used_furniture(scene_path: Path) -> List[dict]:
    with open(scene_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    furniture_by_uid = {
        entry.get("uid"): entry
        for entry in data.get("furniture", [])
        if entry.get("uid")
    }

    used_entries = []
    seen_uids = set()
    for room in data.get("scene", {}).get("room", []):
        for child in room.get("children", []):
            instance_id = child.get("instanceid", "")
            if "furniture" not in instance_id:
                continue
            uid = child.get("ref")
            if not uid or uid in seen_uids:
                continue
            seen_uids.add(uid)
            furniture = furniture_by_uid.get(uid)
            if not furniture:
                continue
            used_entries.append(
                {
                    "scene_json": scene_path.name,
                    "uid": uid,
                    "jid": furniture.get("jid", ""),
                    "category": resolve_category(furniture),
                }
            )
    return used_entries


def sample_candidates(args) -> List[dict]:
    all_scenes = list_scene_jsons(args.front_json_dir)
    if not all_scenes:
        raise FileNotFoundError(f"No .json files found in {args.front_json_dir}")

    rng = random.Random(None if args.seed == 0 else args.seed)
    chosen_scenes = rng.sample(all_scenes, min(args.num_scenes, len(all_scenes)))

    sampled = []
    seen_jids = set()
    for scene_path in chosen_scenes:
        scene_candidates = extract_used_furniture(scene_path)
        rng.shuffle(scene_candidates)
        for candidate in scene_candidates:
            if len(sampled) >= args.max_total_models:
                break
            jid = candidate["jid"]
            if not jid:
                continue
            if not args.allow_duplicate_jids and jid in seen_jids:
                continue
            sampled.append(candidate)
            seen_jids.add(jid)
            if sum(1 for row in sampled if row["scene_json"] == scene_path.name) >= args.max_furniture_per_scene:
                break
        if len(sampled) >= args.max_total_models:
            break
    return sampled


def inspect_mtl(mtl_path: Path) -> dict:
    report = {
        "mtl_exists": mtl_path.exists(),
        "mtl_newmtl_count": 0,
        "mtl_has_map_ka": False,
        "mtl_has_map_kd": False,
        "mtl_has_tf": False,
        "mtl_has_tr": False,
    }
    if not mtl_path.exists():
        return report

    with open(mtl_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("newmtl "):
                report["mtl_newmtl_count"] += 1
            elif stripped.startswith("map_Ka "):
                report["mtl_has_map_ka"] = True
            elif stripped.startswith("map_Kd "):
                report["mtl_has_map_kd"] = True
            elif stripped.startswith("Tf "):
                report["mtl_has_tf"] = True
            elif stripped.startswith("Tr "):
                report["mtl_has_tr"] = True
    return report


def inspect_imported_objects(imported_objects: List) -> dict:
    report = {
        "imported_object_count": len(imported_objects),
        "objects_without_material_slots": 0,
        "objects_with_none_slots": 0,
        "material_slot_count": 0,
        "none_material_slot_count": 0,
        "materials_without_principled": 0,
        "materials_with_multiple_principled": 0,
        "materials_without_teximage": 0,
        "materials_with_unlinked_base_color": 0,
        "materials_with_non_teximage_base_color": 0,
        "teximage_node_count": 0,
        "teximage_nodes_without_image": 0,
    }

    for obj in imported_objects:
        materials = obj.get_materials()
        if not materials:
            report["objects_without_material_slots"] += 1
            continue

        none_slots = sum(mat is None for mat in materials)
        if none_slots:
            report["objects_with_none_slots"] += 1
            report["none_material_slot_count"] += none_slots

        report["material_slot_count"] += len(materials)

        for mat in materials:
            if mat is None:
                continue

            principled_nodes = mat.get_nodes_with_type("BsdfPrincipled")
            teximage_nodes = mat.get_nodes_with_type("ShaderNodeTexImage")
            report["teximage_node_count"] += len(teximage_nodes)
            report["teximage_nodes_without_image"] += sum(
                1 for node in teximage_nodes if getattr(node, "image", None) is None
            )

            if not teximage_nodes:
                report["materials_without_teximage"] += 1

            if len(principled_nodes) == 0:
                report["materials_without_principled"] += 1
                continue
            if len(principled_nodes) > 1:
                report["materials_with_multiple_principled"] += 1
                continue

            principled = principled_nodes[0]
            links = principled.inputs["Base Color"].links
            if not links:
                report["materials_with_unlinked_base_color"] += 1
                continue
            if "TexImage" not in links[0].from_node.bl_idname:
                report["materials_with_non_teximage_base_color"] += 1

    return report


def cleanup_scene():
    bproc.clean_up(clean_up_camera=False)


def build_issue_codes(row: dict) -> List[str]:
    issues = []
    if not row["obj_exists"]:
        issues.append("missing_raw_model_obj")
    if not row["texture_exists"]:
        issues.append("missing_texture_png")
    if not row["mtl_exists"]:
        issues.append("missing_raw_model_mtl")
    if row["mtl_has_map_ka"] and not row["mtl_has_map_kd"]:
        issues.append("mtl_map_ka_without_map_kd")
    if row["objects_without_material_slots"] > 0:
        issues.append("no_material_slots")
    if row["none_material_slot_count"] > 0:
        issues.append("none_material_slots")
    if row["materials_without_principled"] > 0:
        issues.append("no_principled")
    if row["materials_with_multiple_principled"] > 0:
        issues.append("multiple_principled")
    if row["materials_without_teximage"] > 0:
        issues.append("no_teximage")
    if row["materials_with_unlinked_base_color"] > 0:
        issues.append("base_color_unlinked")
    if row["materials_with_non_teximage_base_color"] > 0:
        issues.append("base_color_not_teximage")
    if row["teximage_nodes_without_image"] > 0:
        issues.append("teximage_without_image")
    return issues


def inspect_candidate(candidate: dict, future_model_dir: str) -> dict:
    jid = candidate["jid"]
    folder_path = Path(future_model_dir) / jid
    obj_path = folder_path / "raw_model.obj"
    mtl_path = folder_path / "raw_model.mtl"
    texture_path = folder_path / "texture.png"

    row = {
        "scene_json": candidate["scene_json"],
        "uid": candidate["uid"],
        "jid": jid,
        "category": candidate["category"],
        "obj_exists": obj_path.exists(),
        "texture_exists": texture_path.exists(),
        "status": "not_imported",
        "error": "",
    }
    row.update(inspect_mtl(mtl_path))

    import_report = {
        "imported_object_count": 0,
        "objects_without_material_slots": 0,
        "objects_with_none_slots": 0,
        "material_slot_count": 0,
        "none_material_slot_count": 0,
        "materials_without_principled": 0,
        "materials_with_multiple_principled": 0,
        "materials_without_teximage": 0,
        "materials_with_unlinked_base_color": 0,
        "materials_with_non_teximage_base_color": 0,
        "teximage_node_count": 0,
        "teximage_nodes_without_image": 0,
    }
    row.update(import_report)

    if not obj_path.exists():
        row["status"] = "missing_obj"
        row["issue_codes"] = ",".join(build_issue_codes(row))
        return row

    try:
        imported_objects = bproc.loader.load_obj(str(obj_path))
        row.update(inspect_imported_objects(imported_objects))
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "import_error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_scene()

    row["issue_codes"] = ",".join(build_issue_codes(row))
    if row["status"] == "ok" and row["issue_codes"]:
        row["status"] = "issues_found"
    return row


def write_csv(rows: List[dict], csv_path: Path):
    fieldnames = [
        "scene_json",
        "uid",
        "jid",
        "category",
        "status",
        "error",
        "obj_exists",
        "texture_exists",
        "mtl_exists",
        "mtl_newmtl_count",
        "mtl_has_map_ka",
        "mtl_has_map_kd",
        "mtl_has_tf",
        "mtl_has_tr",
        "imported_object_count",
        "objects_without_material_slots",
        "objects_with_none_slots",
        "material_slot_count",
        "none_material_slot_count",
        "materials_without_principled",
        "materials_with_multiple_principled",
        "materials_without_teximage",
        "materials_with_unlinked_base_color",
        "materials_with_non_teximage_base_color",
        "teximage_node_count",
        "teximage_nodes_without_image",
        "issue_codes",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: List[dict], args) -> dict:
    issue_counter = Counter()
    status_counter = Counter(row["status"] for row in rows)
    for row in rows:
        for code in row["issue_codes"].split(","):
            if code:
                issue_counter[code] += 1

    return {
        "front_json_dir": args.front_json_dir,
        "future_model_dir": args.future_model_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "num_scenes_requested": args.num_scenes,
        "max_furniture_per_scene": args.max_furniture_per_scene,
        "max_total_models": args.max_total_models,
        "allow_duplicate_jids": args.allow_duplicate_jids,
        "num_rows": len(rows),
        "status_counts": dict(status_counter),
        "issue_counts": dict(issue_counter),
        "scene_jsons": sorted({row["scene_json"] for row in rows}),
    }


def main():
    args = parse_args()
    validate_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    sampled_candidates = sample_candidates(args)
    if not sampled_candidates:
        raise RuntimeError("Sampling produced no furniture candidates.")

    bproc.init()

    rows = []
    for index, candidate in enumerate(sampled_candidates, start=1):
        row = inspect_candidate(candidate, args.future_model_dir)
        rows.append(row)
        print(
            f"[{index:02d}/{len(sampled_candidates):02d}] "
            f"{row['scene_json']} | {row['jid']} | {row['status']} | {row['issue_codes']}"
        )

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "furniture_material_audit.csv"
    summary_path = output_dir / "furniture_material_audit_summary.json"

    write_csv(rows, csv_path)
    summary = build_summary(rows, args)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(f"Wrote CSV report to: {csv_path}")
    print(f"Wrote summary report to: {summary_path}")


if __name__ == "__main__":
    main()
