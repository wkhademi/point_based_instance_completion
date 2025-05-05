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


def get_predicted_meshes(data, pred_dir):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))

    for f in pred_files:
        l = os.path.basename(f).split("_")
        scene_name = l[0] + "_" + l[1] + "_" + l[2]

        # cache file name
        data[scene_name].append(f)

    return data


def load_scan(gt_dir, scene_name):
    l = scene_name.split("_")
    scene_id = l[0] + "_" + l[1]
    partial_id = l[2]

    json_file = os.path.join(gt_dir, "json_files", f"{scene_id}.json")
    json_data = read_json(json_file)
    instances = json_data[scene_id]["instances"]

    partial_scan_file = os.path.join(
        gt_dir, "scenes", scene_id, "partials", f"partial_point_cloud_{str(partial_id)}.npz"
    )
    scan_data = np.load(partial_scan_file, allow_pickle=True)

    info_gts = []            
    for instance_id in scan_data["arr_0"].item():
        if int(instance_id) == -1:  # ignore floor/walls
            continue

        # get coordinates of object from partial scan
        object = scan_data["arr_0"].item()[instance_id]
        object_xyz = object[:, :3]

        # get semantic class id
        instance = instances[instance_id]
        category_id = instance["category_id"]
        category_label = ShapeNetIDMap[category_id[1:]]
        label_idx = CAD_labels.index(category_label)

        info_gts.append((label_idx, object_xyz))

    return info_gts


def load_pred_mesh(f):
    mesh = trimesh.load(f, process=False)
    label = extract_label(os.path.basename(f))
    score = extract_score(os.path.basename(f))
    
    return (label, mesh, score)


def eval(gt_dir, pred_dir, threshs=[0.5, 0.75]):
    log_file = open(f'eval_log_pcr.txt', 'a')

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]

    test_scans = read_txt(os.path.join(gt_dir, "test_scene_ids.txt"))
    
    data = {}
    for scene_name in test_scans:
        for partial_id in range(2):
            data[f"{scene_name}_{str(partial_id)}"] = []

    data = get_predicted_meshes(data, pred_dir)

    scene_names = data.keys()

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):

        # gt points
        info_mesh_gts = load_scan(gt_dir, scene_name)

        # pred mesh
        pred_files_scene = data[scene_name]
        info_mesh_preds= []
        for f in pred_files_scene:
            info_mesh_preds.append(load_pred_mesh(f))

        # record
        for calc in ap_calculator_list:
            calc.step([info_mesh_preds], [info_mesh_gts])

        print(f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(info_mesh_gts)},  #pred = {len(info_mesh_preds)}')

    # compute metrics
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- thresh = {threshs[i]} -----')
        print(f'----- thresh = {threshs[i]} -----', file=log_file)
        metrics_dict = calc.compute_metrics()
        for k, v in metrics_dict.items():
            print(f"{k: <50}: {v}")
            print(f"{k: <50}: {v}", file=log_file)
    
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--pred_dir', type=str)
    args = parser.parse_args()

    eval(args.data_dir, args.pred_dir)