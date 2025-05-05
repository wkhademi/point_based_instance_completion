import os
import sys
sys.path.insert(1, os.path.realpath(os.path.curdir))

import glob
import trimesh
import argparse
import numpy as np
import point_cloud_utils as pcu

from evaluation.scene_completion_metrics import *
from utils.io_utils import load_mesh
from utils.misc_utils import *


def load_gt_shapenet_model(model_annotation, shapenet_dir):
    obj_id = model_annotation["cad_id"]
    cat_id = model_annotation["category_id"]
    cad_file = os.path.join(shapenet_dir, cat_id, obj_id, "model.obj")
    object_mesh = load_mesh(cad_file)

    # return watertight mesh to original ShapeNet normalization
    vertices = object_mesh.vertices
    centroid = np.mean(vertices, axis=0)[None, :]
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    diag = np.linalg.norm(bbmax - bbmin)
    vertices = (vertices - centroid) / diag
    object_mesh.vertices = vertices

    # transform from canonical to world coordinates
    t = model_annotation["gt_translation_c2w"]
    q = model_annotation["gt_rotation_quat_wxyz_c2w"]
    s = model_annotation["gt_scale_c2w"]
    M = make_M_from_tqs(t, q, s)
    object_mesh = object_mesh.apply_transform(M)

    return object_mesh


def load_gt_scene(scene_id, instance_ids, data_dir):
    gt_json_file = os.path.join(data_dir, f"json_files/{scene_id}.json")
    scene_annotation_dict = read_json(gt_json_file)
    scene_annotations = scene_annotation_dict[scene_id]

    background = load_mesh(os.path.join(data_dir, f"scenes/{scene_id}/background_mesh.ply"))
    scene_mesh = [background]

    scene_point_cloud = []
    for idx, model_annotation in scene_annotations["instances"].items():
        object_mesh = load_gt_shapenet_model(model_annotation, os.path.join(data_dir, "shapenet_meshes"))
        scene_mesh.append(object_mesh)

        if idx in instance_ids:
            object_point_cloud = np.array(object_mesh.sample(2048), dtype=np.float32)
            scene_point_cloud.append(object_point_cloud)
    scene_point_cloud = np.concatenate(scene_point_cloud, axis=0)

    return scene_mesh, scene_point_cloud


def load_partial_scene(scene_id, partial_id, data_dir):
    partial_path = os.path.join(data_dir, f"scenes/{scene_id}/partials/partial_point_cloud_{partial_id}.npz")
    partial_data = np.load(partial_path, allow_pickle=True)
    instance_ids = list(partial_data["arr_0"].item())

    partial = []
    for key in partial_data["arr_0"].item():
        if int(key) == -1:  # ignore floor/walls
            continue

        partial_object = partial_data["arr_0"].item()[key]
        partial_object_pc = partial_object[:, :3]
        partial.append(partial_object_pc)

    partial = np.concatenate(partial, axis=0).astype(np.float32)

    return partial, instance_ids


def load_completion_scene(pred_dir, scene_id, partial_id, prediction_file):
    prediction_path = os.path.join(pred_dir, prediction_file)
    prediction_data = np.load(prediction_path)
    prediction = prediction_data["xyz"]

    mesh_dir = os.path.join(os.path.dirname(pred_dir), "reconstructions")
    pred_files = glob.glob(f"{mesh_dir}/{scene_id}_{partial_id}_*.ply")
    pred_files = sorted(pred_files, key=lambda x: int(os.path.basename(x).split("_")[3][:-4]))

    mesh_prediction = []
    for pred_file in pred_files:
        mesh = trimesh.load(pred_file, process=False)
        vw, fw = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, 5_000)
        mesh = trimesh.Trimesh(vw, fw)
        mesh_prediction.append(mesh)

    return prediction, mesh_prediction


def evaluate(data_dir, pred_dir):
    partial_files = os.listdir(os.path.join(os.path.dirname(pred_dir), "partials"))
    partial_files.sort()
    prediction_files = os.listdir(pred_dir)
    prediction_files.sort()

    avg_cd = []
    avg_one_side_cd = []
    avg_uhd = []
    avg_collision = []
    avg_num_collisions = []

    for partial_file, prediction_file in zip(partial_files, prediction_files):
        scene_id = prediction_file.split(".")[0][:-2]
        partial_id = partial_file[-5]

        # load partial input
        partial, instance_ids = load_partial_scene(scene_id, partial_id, data_dir)

        # load predicted completion
        prediction, mesh_prediction = load_completion_scene(pred_dir, scene_id, partial_id, prediction_file)

        # load ground truth scene
        gt_meshes, gt = load_gt_scene(scene_id, instance_ids, data_dir)

        # chamfer distance (completion metric)
        cd = 1e3 * chamfer_distance(prediction, gt)
        avg_cd.append(cd)

        # one-sided chamfer distance (partial reconstruction metric)
        one_side_cd = 1e3 * one_sided_chamfer_distance(prediction, partial)
        avg_one_side_cd.append(one_side_cd)

        # unidirectional hausdorff distance (partial reconstruction metric)
        uhd = 1e3 * unidirectional_hausdorff_distance(prediction, partial)
        avg_uhd.append(uhd)

        # collision distance (COL) and percent of points in collison (%COL)
        collision_penalty, num_collisions = collision(prediction, mesh_prediction, gt_meshes[0])
        collision_penalty *= 1e4
        num_collisions *= 100
        avg_collision.append(collision_penalty)
        avg_num_collisions.append(num_collisions)

        print(cd, one_side_cd, uhd, collision_penalty, num_collisions)

    print("CD: ", np.mean(avg_cd))
    print("One sided CD: ", np.mean(avg_one_side_cd))
    print("UHD: ", np.mean(avg_uhd))
    print("Pred Collision: ", np.mean(avg_collision))
    print("# of Pred Collisions: ", np.mean(avg_num_collisions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    args = parser.parse_args()

    evaluate(args.data_dir, args.pred_dir)