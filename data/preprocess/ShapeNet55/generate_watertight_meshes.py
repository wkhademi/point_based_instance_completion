import os
import json
import trimesh
import mesh2sdf
import argparse
import numpy as np

from itertools import repeat
from multiprocessing.pool import Pool


def process_mesh(model_id, category_id, data_dir, output_dir):
    """
    Fix a non-watertight non-manifold mesh to be watertight.
    """
    nonwatertight_mesh_path = os.path.join(
        data_dir, category_id, model_id, "models/model_normalized.obj"
    )
    watertight_mesh_path = os.path.join(
        output_dir, category_id, model_id, "model.obj"
    )

    save_dir = os.path.join(output_dir, category_id, model_id)
    os.makedirs(save_dir, exist_ok=True)

    # ShapeNet specific parameters
    mesh_scale = 0.75
    size = 128
    level = 2 / size

    # load mesh
    mesh = trimesh.load(nonwatertight_mesh_path, force="mesh")

    # normalize mesh to be within [-1, 1] cube
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    vertices -= center
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices *= scale

    # fix mesh to be watertight
    _, mesh = mesh2sdf.compute(
        vertices,
        mesh.faces,
        size,
        fix=True,
        level=level,
        return_mesh=True,
    )

    # save watertight mesh
    mesh.export(watertight_mesh_path)


def process_dataset(args):
    # ensure path to data exists
    assert os.path.isdir(args.data_dir), "data_dir must be a path to an existing data directory"

    # create output dir to store processed data in
    os.makedirs(args.output_dir, exist_ok=True)

    # get list of category directories
    category_ids = [
        category_id for category_id in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, category_id))
    ]

    # process meshes
    with Pool(processes=args.num_processes) as p:
        for category_id in category_ids:
            category_dir = os.path.join(args.data_dir, category_id)

            model_ids = os.listdir(category_dir)

            # skip models for watertight meshes have already been generated
            model_ids = [
                model_id for model_id in model_ids
                if not os.path.exists(
                    os.path.join(
                        args.output_dir, category_id, model_id, "model.obj"
                    )
                )
            ]
            if len(model_ids) == 0:
                continue

            print(f"Processing category id: {category_id}")

            p.starmap(
                process_mesh,
                zip(
                    model_ids,
                    repeat(category_id),
                    repeat(args.data_dir),
                    repeat(args.output_dir),
                ),
            )

    # generate JSON containing model ids present in processed dataset
    dataset_dict = {}
    for category_id in os.listdir(args.output_dir):
        if not os.path.isdir(os.path.join(args.output_dir, category_id)):
            continue

        dataset_dict[category_id] = []

        category_dir = os.path.join(args.output_dir, category_id)
        for model_id in os.listdir(category_dir):
            dataset_dict[category_id].append(model_id)

    model_list_path = os.path.join(args.output_dir, "ShapeNet55_model_list.json")
    with open(model_list_path, "w") as out_file:
        json.dump(dataset_dict, out_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--num_processes", type=int, default=4)
    args = parser.parse_args()

    process_dataset(args)
