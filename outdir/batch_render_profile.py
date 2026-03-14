import hashlib
import json
import math
import os
import random
from pathlib import Path

import bpy
import yaml
from mathutils import Vector

CONFIG_PATH = Path(__file__).with_name("render_profile.yaml")


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return (
        raw.get("paths", {}),
        raw.get("material", {}),
        raw.get("logic", {}),
        raw.get("render", {}),
    )


PATHS_CONFIG, MATERIAL_CONFIG, LOGIC_CONFIG, RENDER_CONFIG = _load_config()


def generate_fibonacci_points(n_samples, radius, center_loc, hemisphere=True):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_samples):
        safe_n = n_samples - 1 if n_samples > 1 else 1
        z = 1 - (i / safe_n) if hemisphere else 1 - (i / safe_n) * 2
        radius_at_z = math.sqrt(1 - z * z) * radius
        theta = phi * i
        x = math.cos(theta) * radius_at_z
        y = math.sin(theta) * radius_at_z
        z_world = z * radius
        points.append(Vector((x, y, z_world)) + center_loc)
    return points


def _get_or_create_principled_material(mat_name):
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
    if not mat.use_nodes:
        mat.use_nodes = True
    return mat


def _new_material(mat_name):
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    return mat


def _stable_int_from_string(s):
    h = hashlib.md5(str(s).encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)


def _weighted_choice(rng, items):
    total = 0.0
    for _, weight in items:
        try:
            numeric = float(weight)
        except Exception:
            continue
        if numeric > 0:
            total += numeric
    if total <= 0:
        return None
    pick = rng.random() * total
    acc = 0.0
    for key, weight in items:
        try:
            numeric = float(weight)
        except Exception:
            continue
        if numeric <= 0:
            continue
        acc += numeric
        if pick <= acc:
            return key
    return items[-1][0] if items else None


def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _srgb_to_linear_1(x):
    x = _clamp01(float(x))
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4


def _rgb_srgb_to_linear(rgb):
    try:
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    except Exception:
        r, g, b = 1.0, 1.0, 1.0
    return (_srgb_to_linear_1(r), _srgb_to_linear_1(g), _srgb_to_linear_1(b))


def _hsv_to_rgb(h, s, v):
    h = float(h) % 1.0
    s = _clamp01(float(s))
    v = _clamp01(float(v))
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


def _sample_albedo_linear(rng, *, value_min, value_max, saturation_max):
    h = rng.random()
    s = rng.uniform(0.0, float(saturation_max))
    v = rng.uniform(float(value_min), float(value_max))
    rgb_srgb = _hsv_to_rgb(h, s, v)
    r, g, b = _rgb_srgb_to_linear(rgb_srgb)
    return (float(r), float(g), float(b), 1.0)


def _rand_log_uniform(rng, a, b):
    a = float(a)
    b = float(b)
    if a <= 0 or b <= 0 or b < a:
        return float(a)
    return math.exp(rng.uniform(math.log(a), math.log(b)))


def sample_material_params_for_object(glb_path):
    seed0 = int(MATERIAL_CONFIG.get("seed", 0) or 0)
    seed = (seed0 ^ _stable_int_from_string(os.path.abspath(glb_path))) & 0xFFFFFFFF
    rng = random.Random(seed)

    sampling_mode = (MATERIAL_CONFIG.get("type_sampling", "fixed") or "fixed").strip().lower()
    if sampling_mode == "weighted":
        weights = MATERIAL_CONFIG.get("type_weights") or {}
        allowed = {
            "origin", "mirror", "glass", "glass_clear", "glass_frosted", "glass_tinted",
            "plastic", "ceramic", "metal_painted", "metal_car_paint", "rubber",
        }
        items = [(str(k).strip().lower(), v) for k, v in weights.items() if str(k).strip().lower() in allowed]
        material_type = _weighted_choice(rng, items) or (MATERIAL_CONFIG.get("type", "mirror") or "mirror")
    else:
        material_type = MATERIAL_CONFIG.get("type", "mirror") or "mirror"
    material_type = str(material_type).strip().lower()

    params = {
        "material_type": material_type,
        "randomize": bool(MATERIAL_CONFIG.get("randomize", False)),
        "seed": int(seed),
    }
    if not params["randomize"]:
        return params

    if material_type in ("glass", "glass_clear", "glass_frosted", "glass_tinted"):
        cfg = MATERIAL_CONFIG.get("random_glass") or {}
        params["ior"] = float(rng.uniform(float(cfg.get("ior_min", 1.47)), float(cfg.get("ior_max", 1.53))))
        if material_type == "glass_frosted":
            params["roughness"] = float(rng.uniform(
                float(cfg.get("frosted_roughness_min", 0.10)),
                float(cfg.get("frosted_roughness_max", 0.60)),
            ))
        else:
            params["roughness"] = float(_rand_log_uniform(
                rng,
                float(cfg.get("clear_roughness_min", 0.0005)),
                float(cfg.get("clear_roughness_max", 0.02)),
            ))
        params["distribution"] = None
        use_tint = material_type == "glass_tinted"
        if material_type in ("glass", "glass_clear") and not use_tint:
            use_tint = rng.random() < float(cfg.get("allow_clear_tint_prob", 0.0))
        params["use_tint_volume"] = bool(use_tint)
        if params["use_tint_volume"]:
            if material_type == "glass_tinted":
                dmin = float(cfg.get("tinted_density_min", 0.60))
                dmax = float(cfg.get("tinted_density_max", 1.80))
                base = cfg.get("tinted_base_rgb", (0.72, 0.84, 0.98))
                jitter = float(cfg.get("tinted_jitter", 0.08))
            else:
                dmin = float(cfg.get("clear_tint_density_min", 0.01))
                dmax = float(cfg.get("clear_tint_density_max", 0.08))
                base = cfg.get("clear_tint_base_rgb", (0.92, 0.97, 1.0))
                jitter = float(cfg.get("clear_tint_jitter", 0.03))
            br, bg, bb = float(base[0]), float(base[1]), float(base[2])
            params["tint_color_rgba"] = (
                _clamp01(br + rng.uniform(-jitter, jitter)),
                _clamp01(bg + rng.uniform(-jitter, jitter)),
                _clamp01(bb + rng.uniform(-jitter, jitter)),
                1.0,
            )
            params["tint_density"] = float(rng.uniform(dmin, dmax))

    elif material_type in ("plastic", "ceramic", "metal_painted", "metal_car_paint", "rubber"):
        if material_type == "plastic":
            cfg = MATERIAL_CONFIG.get("random_plastic") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng, value_min=cfg.get("value_min", 0.15), value_max=cfg.get("value_max", 0.95),
                saturation_max=cfg.get("saturation_max", 0.55),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.05), cfg.get("roughness_max", 0.60)))
            params["specular"] = float(rng.uniform(cfg.get("specular_min", 0.35), cfg.get("specular_max", 0.55)))
            params["clearcoat"] = float(rng.uniform(cfg.get("clearcoat_min", 0.0), cfg.get("clearcoat_max", 0.25)))
            params["clearcoat_roughness"] = float(rng.uniform(cfg.get("clearcoat_roughness_min", 0.0), cfg.get("clearcoat_roughness_max", 0.20)))
            params["metallic"] = 0.0
        elif material_type == "ceramic":
            cfg = MATERIAL_CONFIG.get("random_ceramic") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng, value_min=cfg.get("value_min", 0.40), value_max=cfg.get("value_max", 1.00),
                saturation_max=cfg.get("saturation_max", 0.30),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.03), cfg.get("roughness_max", 0.45)))
            params["specular"] = float(rng.uniform(cfg.get("specular_min", 0.45), cfg.get("specular_max", 0.70)))
            params["clearcoat"] = float(rng.uniform(cfg.get("clearcoat_min", 0.0), cfg.get("clearcoat_max", 0.35)))
            params["clearcoat_roughness"] = float(rng.uniform(cfg.get("clearcoat_roughness_min", 0.0), cfg.get("clearcoat_roughness_max", 0.25)))
            params["metallic"] = 0.0
        elif material_type == "metal_painted":
            cfg = MATERIAL_CONFIG.get("random_metal_painted") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng, value_min=cfg.get("value_min", 0.20), value_max=cfg.get("value_max", 0.95),
                saturation_max=cfg.get("saturation_max", 0.70),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.04), cfg.get("roughness_max", 0.35)))
            params["specular"] = float(rng.uniform(cfg.get("specular_min", 0.40), cfg.get("specular_max", 0.60)))
            params["clearcoat"] = float(rng.uniform(cfg.get("clearcoat_min", 0.20), cfg.get("clearcoat_max", 1.00)))
            params["clearcoat_roughness"] = float(rng.uniform(cfg.get("clearcoat_roughness_min", 0.0), cfg.get("clearcoat_roughness_max", 0.15)))
            params["metallic"] = 0.0
        elif material_type == "metal_car_paint":
            cfg = MATERIAL_CONFIG.get("random_metal_car_paint") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng, value_min=cfg.get("value_min", 0.18), value_max=cfg.get("value_max", 0.90),
                saturation_max=cfg.get("saturation_max", 0.75),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.18), cfg.get("roughness_max", 0.40)))
            params["specular"] = 0.5
            params["clearcoat"] = float(rng.uniform(cfg.get("clearcoat_min", 0.80), cfg.get("clearcoat_max", 1.00)))
            params["clearcoat_roughness"] = float(rng.uniform(cfg.get("clearcoat_roughness_min", 0.0), cfg.get("clearcoat_roughness_max", 0.12)))
            params["metallic"] = 1.0
        elif material_type == "rubber":
            cfg = MATERIAL_CONFIG.get("random_rubber") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng, value_min=cfg.get("value_min", 0.02), value_max=cfg.get("value_max", 0.25),
                saturation_max=cfg.get("saturation_max", 0.20),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.60), cfg.get("roughness_max", 0.95)))
            params["specular"] = float(rng.uniform(cfg.get("specular_min", 0.02), cfg.get("specular_max", 0.20)))
            params["metallic"] = 0.0
            params["clearcoat"] = 0.0
            params["clearcoat_roughness"] = 0.0
            params["sheen"] = float(rng.uniform(cfg.get("sheen_min", 0.0), cfg.get("sheen_max", 0.25)))
            params["sheen_tint"] = float(rng.uniform(cfg.get("sheen_tint_min", 0.0), cfg.get("sheen_tint_max", 0.50)))

    return params


def _reset_principled_material_nodes(mat):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (300, 0)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs.get("BSDF"), out_node.inputs.get("Surface"))
    return bsdf, out_node


def _reset_glass_material_nodes(mat):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (300, 0)
    glass = nodes.new(type="ShaderNodeBsdfGlass")
    glass.location = (0, 0)
    links.new(glass.outputs.get("BSDF"), out_node.inputs.get("Surface"))
    return glass, out_node


def apply_material(obj):
    material_params = None
    try:
        raw = obj.get("_material_params_json", None)
        if raw:
            material_params = json.loads(raw)
    except Exception:
        material_params = None

    material_type = None
    if material_params and material_params.get("material_type"):
        material_type = str(material_params.get("material_type")).strip().lower()
    if not material_type:
        material_type = (MATERIAL_CONFIG.get("type", "mirror") or "mirror").strip().lower()
    if material_type == "origin":
        return

    if material_type == "mirror":
        mat = _new_material(f"Auto_Mirror_{obj.name}") if material_params and material_params.get("randomize", False) else _get_or_create_principled_material("Auto_Mirror_Material")
        bsdf, _ = _reset_principled_material_nodes(mat)
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 1.0
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.0
    elif material_type in ("glass", "glass_clear", "glass_frosted", "glass_tinted"):
        mat = _new_material(f"Auto_Glass_{obj.name}") if material_params and material_params.get("randomize", False) else _get_or_create_principled_material("Auto_Glass_Material")
        glass, out_node = _reset_glass_material_nodes(mat)
        if "Color" in glass.inputs:
            glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        if "Roughness" in glass.inputs:
            glass.inputs["Roughness"].default_value = float(material_params["roughness"]) if material_params and "roughness" in material_params else (float(MATERIAL_CONFIG.get("frosted_roughness", 0.2)) if material_type == "glass_frosted" else 0.0)
        if "IOR" in glass.inputs:
            glass.inputs["IOR"].default_value = float(material_params["ior"]) if material_params and "ior" in material_params else float(MATERIAL_CONFIG.get("ior", 1.5))
        if MATERIAL_CONFIG.get("glass_apply_distribution", True):
            dist_key = "glass_distribution_frosted" if material_type == "glass_frosted" else "glass_distribution_clear"
            dist = (MATERIAL_CONFIG.get(dist_key) or "").strip().upper()
            if dist and hasattr(glass, "distribution"):
                try:
                    glass.distribution = dist
                except Exception:
                    pass
        use_tint = material_type == "glass_tinted"
        if (not use_tint) and material_params:
            use_tint = bool(material_params.get("use_tint_volume", False))
        if use_tint:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            vol = nodes.new(type="ShaderNodeVolumeAbsorption")
            vol.location = (0, -260)
            color = material_params.get("tint_color_rgba") if material_params and "tint_color_rgba" in material_params else MATERIAL_CONFIG.get("tint_color_rgba", (0.85, 0.95, 1.0, 1.0))
            vol.inputs["Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), float(color[3]))
            vol.inputs["Density"].default_value = float(material_params["tint_density"]) if material_params and "tint_density" in material_params else float(MATERIAL_CONFIG.get("tint_density", 0.5))
            links.new(vol.outputs.get("Volume"), out_node.inputs.get("Volume"))
    elif material_type in ("plastic", "ceramic", "metal_painted", "metal_car_paint", "rubber"):
        mat = _new_material(f"Auto_{material_type}_{obj.name}") if material_params and material_params.get("randomize", False) else _get_or_create_principled_material(f"Auto_{material_type}_Material")
        bsdf, _ = _reset_principled_material_nodes(mat)
        if material_params and "base_color_rgba" in material_params and "Base Color" in bsdf.inputs:
            c = material_params["base_color_rgba"]
            bsdf.inputs["Base Color"].default_value = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = float(material_params["metallic"]) if material_params and "metallic" in material_params else 0.0
        if material_params and "roughness" in material_params and "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = float(material_params["roughness"])
        if material_params and "specular" in material_params:
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = float(material_params["specular"])
            elif "Specular" in bsdf.inputs:
                bsdf.inputs["Specular"].default_value = float(material_params["specular"])
        if material_params and "clearcoat" in material_params:
            if "Coat Weight" in bsdf.inputs:
                bsdf.inputs["Coat Weight"].default_value = float(material_params["clearcoat"])
            elif "Clearcoat" in bsdf.inputs:
                bsdf.inputs["Clearcoat"].default_value = float(material_params["clearcoat"])
        if material_params and "clearcoat_roughness" in material_params:
            if "Coat Roughness" in bsdf.inputs:
                bsdf.inputs["Coat Roughness"].default_value = float(material_params["clearcoat_roughness"])
            elif "Clearcoat Roughness" in bsdf.inputs:
                bsdf.inputs["Clearcoat Roughness"].default_value = float(material_params["clearcoat_roughness"])
        if material_type == "rubber":
            if material_params and "sheen" in material_params:
                if "Sheen Weight" in bsdf.inputs:
                    bsdf.inputs["Sheen Weight"].default_value = float(material_params["sheen"])
                elif "Sheen" in bsdf.inputs:
                    bsdf.inputs["Sheen"].default_value = float(material_params["sheen"])
            if material_params and "sheen_tint" in material_params and "Sheen Tint" in bsdf.inputs:
                tint = float(material_params["sheen_tint"])
                socket = bsdf.inputs["Sheen Tint"]
                socket_type = str(getattr(socket, "type", "")).upper()
                if socket_type == "RGBA":
                    socket.default_value = (tint, tint, tint, 1.0)
                elif socket_type == "VECTOR":
                    socket.default_value = (tint, tint, tint)
                else:
                    socket.default_value = tint
    else:
        MATERIAL_CONFIG["type"] = "mirror"
        return apply_material(obj)

    if obj.data.materials:
        obj.data.materials.clear()
    obj.data.materials.append(mat)
    if getattr(obj.data, "polygons", None):
        for poly in obj.data.polygons:
            poly.material_index = 0


def setup_render_settings():
    scene = bpy.context.scene
    scene.render.engine = RENDER_CONFIG["engine"]
    cycles = scene.cycles
    prefs = bpy.context.preferences.addons["cycles"].preferences
    backend = RENDER_CONFIG.get("gpu_backend", "OPTIX")
    try:
        prefs.compute_device_type = backend
    except Exception:
        prefs.compute_device_type = "CUDA"
        backend = "CUDA"
    prefs.get_devices()

    available_gpus = [device for device in prefs.devices if device.type == backend]
    if not available_gpus and backend == "OPTIX":
        available_gpus = [device for device in prefs.devices if device.type == "CUDA"]

    for device in prefs.devices:
        device.use = device.type != "CPU"
    for index, device in enumerate(available_gpus):
        if RENDER_CONFIG.get("gpu_indices") and index not in RENDER_CONFIG["gpu_indices"]:
            device.use = False

    cycles.device = "GPU"
    scene.render.film_transparent = False
    cycles.use_denoising = bool(RENDER_CONFIG.get("use_denoising", False))
    if cycles.use_denoising:
        cycles.denoiser = RENDER_CONFIG.get("denoiser_type", "OPENIMAGEDENOISE")
    if hasattr(cycles, "use_light_tree"):
        cycles.use_light_tree = bool(RENDER_CONFIG.get("use_light_tree", False))
    if hasattr(cycles, "use_adaptive_sampling"):
        cycles.use_adaptive_sampling = False
    cycles.samples = RENDER_CONFIG.get("samples_max", 4096)
    cycles.sample_clamp_direct = RENDER_CONFIG.get("clamp_direct", 0)
    cycles.sample_clamp_indirect = RENDER_CONFIG.get("clamp_indirect", 10.0)
    light_paths = RENDER_CONFIG.get("light_paths", {})
    if light_paths:
        cycles.max_bounces = light_paths.get("max_bounces", 32)
        cycles.diffuse_bounces = light_paths.get("diffuse_bounces", 16)
        cycles.glossy_bounces = light_paths.get("glossy_bounces", 16)
        cycles.transparent_max_bounces = light_paths.get("transparent_max", 32)
        cycles.transmission_bounces = light_paths.get("transmission", 16)
    if hasattr(cycles, "caustics_reflective"):
        cycles.caustics_reflective = bool(RENDER_CONFIG.get("caustics_reflective", True))
    if hasattr(cycles, "caustics_refractive"):
        cycles.caustics_refractive = bool(RENDER_CONFIG.get("caustics_refractive", True))
    if hasattr(cycles, "blur_glossy"):
        cycles.blur_glossy = float(RENDER_CONFIG.get("blur_glossy", 0.0))
    if hasattr(scene.view_settings, "view_transform"):
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    scene.render.resolution_x = RENDER_CONFIG["res_x"]
    scene.render.resolution_y = RENDER_CONFIG["res_y"]
    scene.render.resolution_percentage = RENDER_CONFIG["res_percent"]


def create_smart_camera(anchor_obj):
    scene = bpy.context.scene
    if "UltraCam" in scene.objects:
        bpy.data.objects.remove(scene.objects["UltraCam"], do_unlink=True)
    cam_data = bpy.data.cameras.new(name="UltraCam")
    cam_obj = bpy.data.objects.new(name="UltraCam", object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_data.lens = LOGIC_CONFIG["lens"]
    target_loc = anchor_obj.location if anchor_obj else Vector((0, 0, 0))
    if "FocusTarget" not in scene.objects:
        empty = bpy.data.objects.new("FocusTarget", None)
        scene.collection.objects.link(empty)
    target_obj = scene.objects["FocusTarget"]
    target_obj.location = target_loc
    constraint = cam_obj.constraints.new(type="TRACK_TO")
    constraint.target = target_obj
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    return cam_obj
