import os
import sys
sys.path.insert(1, os.path.realpath(os.path.curdir))

import glob
import trimesh
import argparse
import matplotlib
import numpy as np

from utils.misc_utils import *
from utils.io_utils import load_mesh


def load_partial_scene(scene_id, partial_id, data_dir):
    partial_path = os.path.join(data_dir, f"scenes/{scene_id}/partials/partial_point_cloud_{partial_id}.npz")
    partial_data = np.load(partial_path, allow_pickle=True)
    instance_ids = list(partial_data["arr_0"].item())

    partial = []
    for key in partial_data["arr_0"].item():
        partial_object = partial_data["arr_0"].item()[key]
        partial_object_pc = partial_object[:, :3]
        partial.append(partial_object_pc)
    partial = np.concatenate(partial, axis=0).astype(np.float32)

    return partial, instance_ids


def load_predicted_scene(scene_id, partial_id, pred_dir):
    # load predicted point cloud scene completion
    prediction_path = os.path.join(pred_dir, f"{scene_id}_{partial_id}.npz")
    prediction_data = np.load(prediction_path)
    point_prediction = np.concatenate([prediction_data["xyz"], prediction_data["normal"]], axis=1)

    # mesh reconstruction files
    mesh_dir = os.path.join(os.path.dirname(pred_dir), "reconstructions")
    pred_files = glob.glob(f"{mesh_dir}/{scene_id}_{partial_id}_*.ply")
    pred_files = sorted(pred_files, key=lambda x: int(os.path.basename(x).split("_")[3][:1]))

    # load mesh reconstructions
    mesh_prediction = []
    pred_categories = []
    for pred_file in pred_files:
        mesh = load_mesh(pred_file)
        mesh_prediction.append(mesh)

        label = extract_label(os.path.basename(pred_file))
        if label is not None:
            category = CAD_labels[label]
            pred_categories.append(category)

    return point_prediction, mesh_prediction, pred_categories


def load_gt_shapenet_model(model_annotation, data_dir):
    # load gt mesh
    obj_id = model_annotation["cad_id"]
    cat_id = model_annotation["category_id"]
    category = ShapeNetIDMap[cat_id[1:]]
    cad_file = os.path.join(data_dir, "shapenet_meshes", cat_id, obj_id, "model.obj")
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

    return object_mesh, category


def load_gt_scene(scene_id, instance_ids, data_dir):
    # load gt scene annotation file
    gt_json_file = os.path.join(data_dir, f"json_files/{scene_id}.json")
    scene_annotation_dict = read_json(gt_json_file)
    scene_annotations = scene_annotation_dict[scene_id]

    # load background mesh
    background = load_mesh(os.path.join(data_dir, f"scenes/{scene_id}/background_mesh.ply"))
    background.faces = np.fliplr(background.faces)

    scene_mesh = [background]
    categories = []

    # load gt meshes
    for idx, model_annotation in scene_annotations["instances"].items():
        if idx in instance_ids:
            object_mesh, object_category = load_gt_shapenet_model(model_annotation, data_dir)
            scene_mesh.append(object_mesh)
            categories.append(object_category)

    return scene_mesh, categories


def shade_color(rgb, normal):
    hsv = matplotlib.colors.rgb_to_hsv(rgb.astype(np.float32) / 255.)
    luminance = np.dot(normal.astype(np.float32) / 255., [0.2989, 0.5870, 0.1140])
    hsv[:, 2] = hsv[:, 2] * luminance
    shaded_rgb = (255 * matplotlib.colors.hsv_to_rgb(hsv)).astype(np.uint8)

    return shaded_rgb


def visualize_scene(partial_input, point_prediction, mesh_prediction, mesh_gt, categories):
    palette_cls = (255 * np.array([*sns.color_palette("hls", 8)])).astype(np.uint8)

    # center of scene
    min_bound = np.min(partial_input, axis=0)
    max_bound = np.max(partial_input, axis=0)
    center = (min_bound + max_bound) / 2.

    # partial input
    p = trimesh.points.PointCloud(partial_input - center)
    partial_scene = trimesh.Scene([p])

    # reconstructured meshes
    for idx, mesh in enumerate(mesh_prediction):
        mesh.vertices -= center

        if len(mesh_prediction) != len(categories):
            if idx != 0:
                rgb = np.tile(palette_cls[(idx-1) % 8], (mesh.vertices.shape[0], 1))
                mesh.visual.vertex_colors = rgb
        else:
            rgb = np.tile(palette_cls[idx % 8], (mesh.vertices.shape[0], 1))
            mesh.visual.vertex_colors = rgb

    # gt meshes
    for idx, mesh in enumerate(mesh_gt):
        mesh.vertices -= center

        if idx != 0:
            rgb = np.tile(palette_cls[(idx-1) % 8], (mesh.vertices.shape[0], 1))
            mesh.visual.vertex_colors = rgb

    # point clouds
    pc_size = 2048
    pred_points_scene = []
    pred_normals_scene = []
    for idx in range(len(mesh_prediction)):
        object_pc = point_prediction[idx * pc_size : (idx + 1) * pc_size, :3] - center
        object_surface_normals = point_prediction[idx * pc_size : (idx + 1) * pc_size, 3:]

        rgb = np.tile(palette_cls[idx % 8], (object_pc.shape[0], 1))
        normal = (((object_surface_normals + 1) * 0.5) * 255).astype(np.uint8)
        shaded_color = shade_color(rgb, normal)

        # object point cloud
        object_points = trimesh.points.PointCloud(object_pc, shaded_color)
        pred_points_scene.append(object_points)

        # object surface normals
        object_normals = trimesh.points.PointCloud(object_pc, normal)
        pred_normals_scene.append(object_normals)
    
    completion_scene = trimesh.Scene(pred_points_scene)
    normals_scene = trimesh.Scene(pred_normals_scene)
    pred_mesh_scene = trimesh.Scene(mesh_prediction+[p])
    gt_mesh_scene = trimesh.Scene(mesh_gt+[p])

    # visualize scene
    partial_scene.show(line_settings={'point_size': 5})
    completion_scene.show(line_settings={'point_size': 5})
    normals_scene.show(line_settings={'point_size': 5})
    pred_mesh_scene.show(
        line_settings={'point_size': 5},
        flags={'cull': False}
    )
    gt_mesh_scene.show(
        line_settings={'point_size': 5},
    )


def visualize_scenes(data_dir, pred_dir, scene_id):
    prediction_files = os.listdir(pred_dir)
    prediction_scenes = [prediction_scene[:-4] for prediction_scene in prediction_files]

    if scene_id != "all":
        prediction_scenes = [scene_id]

    for prediction_scene in prediction_scenes:
        s = prediction_scene.split("_")
        scene_id = s[0] + "_" + s[1]
        partial_id = s[2]

        # load partial input
        partial_input, instance_ids = load_partial_scene(scene_id, partial_id, data_dir)

        # load predicted completion
        point_prediction, mesh_prediction, pred_categories = \
            load_predicted_scene(scene_id, partial_id, pred_dir)

        # load ground truth mesh
        mesh_gt, gt_categories = load_gt_scene(scene_id, instance_ids, data_dir)

        # visualize prediction
        visualize_scene(
            partial_input, 
            point_prediction,
            mesh_prediction, 
            mesh_gt, 
            pred_categories if len(pred_categories) != 0 else gt_categories,
        )
        print(scene_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--scene_id', type=str, default='all')
    args = parser.parse_args()

    visualize_scenes(args.data_dir, args.pred_dir, args.scene_id)