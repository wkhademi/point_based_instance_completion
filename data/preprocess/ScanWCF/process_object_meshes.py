import os
import sys
sys.path.insert(1, os.path.realpath(os.path.curdir))

import json
import argparse

from itertools import repeat
from multiprocessing.pool import Pool

from data.preprocess.ShapeNet55.generate_watertight_meshes import process_mesh


def process_meshes(args):
    shapenet_dir = args.shapenet_dir
    scanwcf_dir = args.scanwcf_dir
    
    # find all the ShapeNet models used in scenes
    json_dir = os.path.join(scanwcf_dir, "json_files")
    json_files = os.listdir(json_dir)
    shapenet_models = set()
    for json_file in json_files:
        with open(os.path.join(json_dir, json_file), 'r') as infile:
            scene_json = json.load(infile)
        
        scene_name = json_file[:-5]
        object_instance_annotations = scene_json[scene_name]["instances"]

        for k, v in object_instance_annotations.items():
            shapenet_models.add((v["category_id"], v["cad_id"]))

    # generate watertight meshes
    save_dir = os.path.join(scanwcf_dir, "shapenet_meshes")
    category_ids = [item[0] for item in shapenet_models]
    model_ids = [item[1] for item in shapenet_models]
    with Pool(processes=args.num_processes) as p:
        p.starmap(
            process_mesh,
            zip(
                model_ids,
                category_ids,
                repeat(shapenet_dir),
                repeat(save_dir),
            ),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_dir", required=True, type=str)
    parser.add_argument("--scanwcf_dir", required=True, type=str)
    parser.add_argument("--num_processes", type=int, default=16)
    args = parser.parse_args()

    process_meshes(args)