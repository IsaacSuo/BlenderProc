import blenderproc as bproc

import argparse
import os
from pathlib import Path

import bpy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a 3D-FRONT scene and save it as a packed .blend file for local inspection."
    )
    parser.add_argument("front_json", help="Path to a single 3D-FRONT json file.")
    parser.add_argument("future_model_dir", help="Path to the 3D-FUTURE-model directory.")
    parser.add_argument("front_texture_dir", help="Path to the 3D-FRONT-texture directory.")
    parser.add_argument("output_blend", help="Path to the output .blend file.")
    parser.add_argument(
        "--lamp-light-strength",
        type=float,
        default=7.0,
        help="Built-in lamp emission strength used by load_front3d.",
    )
    parser.add_argument(
        "--ceiling-light-strength",
        type=float,
        default=0.8,
        help="Built-in ceiling emission strength used by load_front3d.",
    )
    parser.add_argument(
        "--skip-pack",
        action="store_true",
        help="Do not pack external resources into the output .blend.",
    )
    return parser.parse_args()


def validate_paths(args):
    for path in [args.front_json, args.future_model_dir, args.front_texture_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    output_parent = os.path.dirname(os.path.abspath(args.output_blend))
    os.makedirs(output_parent, exist_ok=True)


def main():
    args = parse_args()
    validate_paths(args)

    bproc.init()

    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    bproc.loader.load_front3d(
        json_path=args.front_json,
        future_model_path=args.future_model_dir,
        front_3D_texture_path=args.front_texture_dir,
        label_mapping=mapping,
        ceiling_light_strength=args.ceiling_light_strength,
        lamp_light_strength=args.lamp_light_strength,
    )

    if not args.skip_pack:
        bpy.ops.file.pack_all()

    output_path = os.path.abspath(args.output_blend)
    bpy.ops.wm.save_as_mainfile(filepath=output_path)

    print(f"Saved blend: {output_path}")
    print(f"Source json: {os.path.abspath(args.front_json)}")
    print(f"Packed resources: {not args.skip_pack}")


if __name__ == "__main__":
    main()
