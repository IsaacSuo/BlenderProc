"""Lamp lighting module: creates real Blender lights for 3D-FRONT lamp objects.

Usage in front3d_place_and_render.py:
    import lamp_lighting
    lamp_lighting.apply_lighting_to_scene(room_objs, paths["front_json"])
"""

import json
import os
from pathlib import Path

import bpy
import numpy as np
import yaml
from mathutils import Vector

from blenderproc.python.types.LightUtility import Light
from blenderproc.python.types.MeshObjectUtility import MeshObject

CONFIG_PATH = Path(__file__).with_name("lamp_config.yaml")

# category_id whitelist for lamp objects
LAMP_CATEGORY_IDS = {8, 19, 32}  # ceiling lamp / lighting / pendant lamp

# Cache for loaded config
_config_cache = None


def _load_lamp_config(force_reload=False):
    """Load lamp_config.yaml with caching."""
    global _config_cache
    if _config_cache is not None and not force_reload:
        return _config_cache
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config_cache = yaml.safe_load(f) or {}
    return _config_cache


def _normalize_category(name):
    """'Pendant Lamp' -> 'pendant_lamp'"""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _resolve_params(jid, category):
    """Merge config: models[jid] > categories[category] > defaults."""
    config = _load_lamp_config()
    params = dict(config.get("defaults", {}))
    if category:
        cat_key = _normalize_category(category)
        cat_params = config.get("categories", {}).get(cat_key, {})
        params.update(cat_params)
    if jid:
        model_params = config.get("models", {})
        if model_params and jid in model_params and model_params[jid]:
            params.update(model_params[jid])
    return params


def _find_lamp_objects(room_objs):
    """Find lamp objects by category_id whitelist or name containing 'lamp'."""
    lamps = []
    for obj in room_objs:
        try:
            cat_id = obj.get_cp("category_id")
        except Exception:
            cat_id = None
        if cat_id in LAMP_CATEGORY_IDS:
            lamps.append(obj)
        elif cat_id is None:
            try:
                name = obj.get_name().lower()
                if "lamp" in name:
                    lamps.append(obj)
            except Exception:
                continue
    return lamps


def _build_uid_to_jid_map(front_json_path):
    """Read the 3D-FRONT JSON furniture array, return {uid: {jid, category}}."""
    with open(front_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for ele in data.get("furniture", []):
        uid = ele.get("uid")
        if uid:
            category = ele.get("category", ele.get("title", ""))
            if "/" in category:
                category = category.split("/")[0]
            mapping[uid] = {
                "jid": ele.get("jid", ""),
                "category": category,
            }
    return mapping


def _apply_emission(mesh_obj, strength):
    """Set or remove emissive materials on a mesh object."""
    for i, mat in enumerate(mesh_obj.get_materials()):
        if mat is None:
            continue
        if mat.get_users() > 1:
            mat = mat.duplicate()
            mesh_obj.set_material(i, mat)
        mat.remove_emissive()
        if strength > 0:
            mat.make_emissive(strength)


def create_light_for_lamp(mesh_obj, params, lamp_info=None):
    """Create a Blender light for a single lamp mesh object.

    Args:
        mesh_obj: The lamp MeshObject.
        params: Resolved parameter dict from config.
        lamp_info: Optional dict with 'jid' and 'category' for naming.

    Returns:
        Light object created.
    """
    light_type = params.get("light_type", "POINT")
    energy = float(params.get("energy", 100.0))
    color = params.get("color", [1.0, 0.95, 0.9])
    radius = float(params.get("radius", 0.1))
    offset = params.get("offset", [0.0, -0.15, 0.0])
    keep_emission = params.get("keep_emission", True)
    emission_strength = float(params.get("emission_strength", 3.0))

    # Compute light position: world bbox center + rotated offset
    bbox = mesh_obj.get_bound_box(local_coords=False)
    bbox_center = np.mean(bbox, axis=0)

    # Transform offset by mesh rotation
    rot_mat = mesh_obj.blender_obj.matrix_world.to_3x3()
    offset_vec = rot_mat @ Vector(offset)
    light_pos = Vector(bbox_center) + offset_vec

    # Create the light
    jid = (lamp_info or {}).get("jid", "unknown")
    light_name = f"LampLight_{jid[:8]}"
    light = Light(light_type=light_type, name=light_name)
    light.set_location(list(light_pos))
    light.set_energy(energy)
    light.set_color(color[:3])
    light.set_radius(radius)

    # AREA light: set size
    if light_type == "AREA":
        area_size = params.get("area_size", [0.4, 0.4])
        light.blender_obj.data.shape = 'RECTANGLE'
        light.blender_obj.data.size = float(area_size[0])
        light.blender_obj.data.size_y = float(area_size[1])

    # Tag for cleanup
    light.blender_obj["lamp_module"] = True

    # Handle emission on the mesh itself
    if keep_emission:
        _apply_emission(mesh_obj, emission_strength)
    else:
        _apply_emission(mesh_obj, 0)

    return light


def apply_lighting_to_scene(room_objs, front_json_path):
    """Main entry point: find lamps in room, create real lights for each.

    Args:
        room_objs: List of MeshObject from load_front3d().
        front_json_path: Path to the 3D-FRONT JSON file.

    Returns:
        List of Light objects created.
    """
    uid_map = _build_uid_to_jid_map(front_json_path)
    lamps = _find_lamp_objects(room_objs)
    lights = []

    for lamp_obj in lamps:
        try:
            uid = lamp_obj.get_cp("uid")
        except Exception:
            uid = None

        lamp_info = uid_map.get(uid, {}) if uid else {}
        jid = lamp_info.get("jid", "")
        category = lamp_info.get("category", "")

        params = _resolve_params(jid, category)
        light = create_light_for_lamp(lamp_obj, params, lamp_info)
        lights.append(light)

    if lights:
        print(f"[lamp_lighting] Created {len(lights)} real lights for {len(lamps)} lamp objects")
    return lights


def remove_lamp_lights():
    """Remove all lights created by this module (identified by lamp_module tag)."""
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.get("lamp_module") and obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    if removed:
        print(f"[lamp_lighting] Removed {removed} lamp lights")
