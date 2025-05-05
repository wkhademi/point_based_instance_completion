# offline mesh IoU eval

import os
import sys
sys.path.insert(1, os.path.realpath(os.path.curdir))

import glob
import trimesh
import argparse
import numpy as np

from metrics import *
from utils.misc_utils import *


def get_gt_meshes(data, gt_dir):
    json_dir = os.path.join(gt_dir, "json_files")
    json_files = os.listdir(json_dir)

    test_scans = read_txt(os.path.join(gt_dir, "test_scene_ids.txt"))

    for json_file in json_files:
        if json_file[:-5] not in test_scans:  # skip training scenes
            continue

        json_file = os.path.join(json_dir, json_file)
        json_data = read_json(json_file)

        scene_name = list(json_data.keys())[0]

        for partial_id in range(2):
            if f"{scene_name}_{str(partial_id)}" not in data: data[f"{scene_name}_{str(partial_id)}"] = {}
            if 'gt' not in data[f"{scene_name}_{str(partial_id)}"]: data[f"{scene_name}_{str(partial_id)}"]['gt'] = []
            if 'pred' not in data[f"{scene_name}_{str(partial_id)}"]: data[f"{scene_name}_{str(partial_id)}"]['pred'] = []

            partial_scan_file = os.path.join(
                gt_dir, "scenes", scene_name, "partials", f"partial_point_cloud_{str(partial_id)}.npz"
            )
            scan_data = np.load(partial_scan_file, allow_pickle=True)
            scene_inst_ids = list(scan_data["arr_0"].item())
            for instance_id, instance in json_data[scene_name]["instances"].items():
                if instance_id not in scene_inst_ids:  # skip objects not present in partial scan
                    continue

                category_id = instance["category_id"]
                cad_id = instance["cad_id"]
                f = os.path.join(gt_dir, "shapenet_meshes", category_id, cad_id, "model.obj")

                t = instance["gt_translation_c2w"]
                q = instance["gt_rotation_quat_wxyz_c2w"]
                s = instance["gt_scale_c2w"]
                M = make_M_from_tqs(t, q, s)

                # cache file name and object pose
                data[f"{scene_name}_{str(partial_id)}"]['gt'].append((f, M))

    return data


def get_predicted_meshes(data, pred_dir):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))

    for f in pred_files:
        l = os.path.basename(f).split("_")
        scene_name = l[0] + "_" + l[1] + "_" + l[2]

        # cache file name
        data[scene_name]['pred'].append(f)

    return data


def load_gt_mesh(f, M):
    # load gt mesh
    mesh = trimesh.load(f, process=False)

    # transform mesh from object coordinates to world coordinates
    vertices = mesh.vertices
    centroid = np.mean(vertices, axis=0)[None, :]
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    diag = np.linalg.norm(bbmax - bbmin)
    vertices = (vertices - centroid) / diag
    mesh.vertices = vertices
    mesh = mesh.apply_transform(M)

    # get category of object instance
    cad_id = os.path.basename(os.path.dirname(os.path.dirname(f)))
    class_name = ShapeNetIDMap[cad_id[1:]]
    label = CAD_labels.index(class_name)

    return (label, mesh)
    

def load_pred_mesh(f):
    mesh = trimesh.load(f, process=False)
    label = extract_label(os.path.basename(f))
    score = extract_score(os.path.basename(f))

    return (label, mesh, score)


def eval(gt_dir, pred_dir, threshs):
    
    log_file = open(f'eval_log_cd.txt', 'a')

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]

    data = {}

    # collect meshes (ply)
    data = get_gt_meshes(data, gt_dir)
    data = get_predicted_meshes(data, pred_dir)

    scene_names = data.keys()

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):

        # load ground truth meshes in scene
        gt_files_scene = data[scene_name]['gt']
        info_mesh_gts = []
        for (f, M) in gt_files_scene:
            info_mesh_gts.append(load_gt_mesh(f, M))

        # load predicted meshes in scene
        pred_files_scene = data[scene_name]['pred']
        info_mesh_preds = []
        for f in pred_files_scene:
            info_mesh_preds.append(load_pred_mesh(f))

        # record
        for calc in ap_calculator_list:
            calc.step(info_mesh_preds, info_mesh_gts)

        print(f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(gt_files_scene)},  #pred = {len(pred_files_scene)}')

    # compute metrics
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- thresh = {threshs[i]} -----')
        print(f'----- thresh = {threshs[i]} -----', file=log_file)
        metrics_dict = calc.compute_metrics()
        for k, v in metrics_dict.items():
            if 'Q_mesh' in k: continue
            if 'mesh' not in k: continue
            print(f"{k: <50}: {v}")
            print(f"{k: <50}: {v}", file=log_file)
    
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--pred_dir', type=str)
    args = parser.parse_args()

    eval(args.data_dir, args.pred_dir, threshs=[0.047, 0.1])