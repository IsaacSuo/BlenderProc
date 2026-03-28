import argparse
import json
import math
import os
import re

import bpy
from mathutils import Vector


FIRST_TIER_SUPPORTS = {
    36: "table",
    68: "dining table",
    3: "tea table",
    81: "desk",
    80: "dressing table",
    72: "corner/side table",
    12: "round end table",
    17: "bed",
    67: "double bed",
    70: "single bed",
    20: "kids bed",
    87: "bunk bed",
    88: "bed frame",
    76: "nightstand",
    56: "tv stand",
    28: "sideboard / side cabinet / console",
    74: "shelf",
    69: "cabinet/shelf/desk",
    48: "cabinet",
    6: "children cabinet",
    47: "kitchen cabinet",
    55: "drawer chest / corner cabinet",
    91: "bookcase / jewelry armoire",
    22: "storage unit",
    23: "media unit",
    84: "wardrobe",
    65: "wine cooler",
}

SECOND_TIER_SUPPORTS = {
    46: "sofa",
    18: "two-seat sofa",
    89: "three-seat / multi-person sofa",
    96: "l-shaped sofa",
    9: "chaise longue sofa",
    10: "lazy sofa",
    16: "armchair",
    50: "chair",
    14: "dining chair",
    78: "lounge chair / book-chair / computer chair",
    71: "classic chinese chair",
    83: "dressing chair",
    94: "barstool",
    35: "pier/stool",
    25: "footstool / sofastool / bed end stool / stool",
    66: "outdoor furniture",
}

FLOOR_SUPPORTS = {
    51: "floor",
}


def parse_args():
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser(
        description="Add ANCHOR objects onto support meshes in a .blend scene and save the result."
    )
    parser.add_argument(
        "--output-blend",
        required=True,
        help="Path to the output .blend file.",
    )
    parser.add_argument(
        "--include-secondary-supports",
        action="store_true",
        help="Also add anchors to sofa/chair/stool-like supports from the outdir whitelist.",
    )
    parser.add_argument(
        "--include-floor",
        action="store_true",
        help="Also add anchors to floor objects.",
    )
    parser.add_argument(
        "--replace-existing-anchors",
        action="store_true",
        help="Remove existing ANCHOR* objects before creating new ones.",
    )
    parser.add_argument(
        "--anchor-display-size",
        type=float,
        default=1.0,
        help="Empty display size for anchors.",
    )
    parser.add_argument(
        "--anchor-scale",
        type=float,
        default=0.5,
        help="Uniform scale applied to anchor empties.",
    )
    parser.add_argument(
        "--surface-angle-deg",
        type=float,
        default=15.0,
        help="Maximum angle from world up when searching for top faces.",
    )
    parser.add_argument(
        "--top-band",
        type=float,
        default=0.03,
        help="World-space Z band used to group top faces into one support patch.",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Extra world-space Z offset applied to created anchors.",
    )
    return parser.parse_args(argv)


def normalized_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def build_support_rules(include_secondary, include_floor):
    supports = dict(FIRST_TIER_SUPPORTS)
    if include_secondary:
        supports.update(SECOND_TIER_SUPPORTS)
    if include_floor:
        supports.update(FLOOR_SUPPORTS)

    keywords = set()
    for label in supports.values():
        clean = normalized_text(label)
        if clean:
            keywords.add(clean)
        for token in clean.split():
            if token not in {"side", "end", "multi", "person", "book", "classic", "chinese"}:
                keywords.add(token)
    keywords.update({"desk", "table"})
    return supports, sorted(keywords)


def is_anchor_name(name):
    return bool(re.match(r"^ANCHOR($|[._].*)", name))


def remove_existing_anchors():
    removed = []
    for obj in list(bpy.data.objects):
        if is_anchor_name(obj.name):
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
    return removed


def get_category_id(obj):
    try:
        return int(obj.get("category_id"))
    except Exception:
        return None


def match_support_object(obj, support_map, keywords):
    if obj.type != "MESH":
        return None
    if is_anchor_name(obj.name):
        return None
    if obj.hide_render:
        return None

    category_id = get_category_id(obj)
    if category_id in support_map:
        return {
            "match_type": "category_id",
            "category_id": category_id,
            "support_label": support_map[category_id],
        }

    name = normalized_text(obj.name)
    for keyword in keywords:
        if keyword and keyword in name:
            return {
                "match_type": "name",
                "category_id": category_id,
                "support_label": keyword,
            }
    return None


def world_bbox_stats(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((
        min(v.x for v in corners),
        min(v.y for v in corners),
        min(v.z for v in corners),
    ))
    max_corner = Vector((
        max(v.x for v in corners),
        max(v.y for v in corners),
        max(v.z for v in corners),
    ))
    dims = max_corner - min_corner
    return min_corner, max_corner, dims


def find_support_surface_point(obj, angle_deg, top_band, z_offset):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    up = Vector((0.0, 0.0, 1.0))
    min_dot = math.cos(math.radians(angle_deg))
    try:
        faces = []
        normal_mat = obj_eval.matrix_world.to_3x3()
        for poly in mesh.polygons:
            world_normal = (normal_mat @ poly.normal).normalized()
            if world_normal.dot(up) < min_dot:
                continue
            center = obj_eval.matrix_world @ poly.center
            faces.append((center, float(poly.area)))

        if faces:
            top_z = max(center.z for center, _ in faces)
            top_faces = [(center, area) for center, area in faces if top_z - center.z <= top_band]
            if top_faces:
                total_weight = sum(area for _, area in top_faces)
                if total_weight <= 0:
                    total_weight = float(len(top_faces))
                x = sum(center.x * (area if area > 0 else 1.0) for center, area in top_faces) / total_weight
                y = sum(center.y * (area if area > 0 else 1.0) for center, area in top_faces) / total_weight
                return Vector((x, y, top_z + z_offset)), "surface_faces"
    finally:
        obj_eval.to_mesh_clear()

    min_corner, max_corner, _ = world_bbox_stats(obj)
    return Vector((
        (min_corner.x + max_corner.x) * 0.5,
        (min_corner.y + max_corner.y) * 0.5,
        max_corner.z + z_offset,
    )), "bbox_top"


def sanitize_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    return cleaned.strip("_") or "Support"


def create_anchor(anchor_collection, support_obj, support_meta, anchor_location, anchor_display_size, anchor_scale, source_method):
    anchor_name = f"ANCHOR_{sanitize_name(support_obj.name)}"
    anchor = bpy.data.objects.new(anchor_name, None)
    anchor.empty_display_type = "CUBE"
    anchor.empty_display_size = float(anchor_display_size)
    anchor.scale = (float(anchor_scale), float(anchor_scale), float(anchor_scale))
    anchor.location = anchor_location
    anchor["anchor_support_object"] = support_obj.name
    if support_meta.get("category_id") is not None:
        anchor["anchor_support_category_id"] = int(support_meta["category_id"])
    anchor["anchor_support_label"] = str(support_meta["support_label"])
    anchor["anchor_match_type"] = str(support_meta["match_type"])
    anchor["anchor_source_method"] = source_method
    anchor_collection.objects.link(anchor)
    return anchor


def get_or_create_anchor_collection():
    scene_collection = bpy.context.scene.collection
    collection = bpy.data.collections.get("ANCHORS")
    if collection is None:
        collection = bpy.data.collections.new("ANCHORS")
        scene_collection.children.link(collection)
    elif collection.name not in {child.name for child in scene_collection.children}:
        scene_collection.children.link(collection)
    return collection


def candidate_sort_key(item):
    obj = item["object"]
    min_corner, max_corner, dims = world_bbox_stats(obj)
    horizontal_area = float(dims.x * dims.y)
    priority = item["priority"]
    return (priority, -horizontal_area, obj.name.lower())


def main():
    args = parse_args()
    support_map, keywords = build_support_rules(args.include_secondary_supports, args.include_floor)

    if args.replace_existing_anchors:
        removed = remove_existing_anchors()
    else:
        removed = []

    candidates = []
    for obj in bpy.data.objects:
        support_meta = match_support_object(obj, support_map, keywords)
        if support_meta is None:
            continue
        category_id = support_meta.get("category_id")
        priority = 0 if category_id in FIRST_TIER_SUPPORTS else 1 if category_id in SECOND_TIER_SUPPORTS else 2
        candidates.append({
            "object": obj,
            "meta": support_meta,
            "priority": priority,
        })

    candidates.sort(key=candidate_sort_key)

    anchor_collection = get_or_create_anchor_collection()
    created = []
    for item in candidates:
        obj = item["object"]
        anchor_location, source_method = find_support_surface_point(
            obj,
            angle_deg=args.surface_angle_deg,
            top_band=args.top_band,
            z_offset=args.z_offset,
        )
        anchor = create_anchor(
            anchor_collection=anchor_collection,
            support_obj=obj,
            support_meta=item["meta"],
            anchor_location=anchor_location,
            anchor_display_size=args.anchor_display_size,
            anchor_scale=args.anchor_scale,
            source_method=source_method,
        )
        created.append({
            "anchor_name": anchor.name,
            "support_object": obj.name,
            "support_label": item["meta"]["support_label"],
            "category_id": item["meta"].get("category_id"),
            "location": [round(float(v), 4) for v in anchor.location],
            "source_method": source_method,
        })

    output_blend = os.path.abspath(args.output_blend)
    os.makedirs(os.path.dirname(output_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)

    summary = {
        "source_blend": bpy.data.filepath,
        "output_blend": output_blend,
        "removed_existing_anchors": removed,
        "anchor_count": len(created),
        "anchors": created,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
