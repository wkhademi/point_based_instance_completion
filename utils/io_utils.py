import os
import array
import Imath
import torch
import OpenEXR
import trimesh
import numpy as np


def load_point_cloud(pc_file):
    """
    Load point cloud from a .ply file.
    """
    mesh_kwargs = trimesh.load_ply(pc_file)
    mesh = trimesh.Trimesh(**mesh_kwargs)
    point_cloud = mesh.vertices

    return point_cloud


def load_mesh(mesh_file):
    """
    Load mesh.
    """
    mesh = trimesh.load(mesh_file, force="mesh", process=False)

    return mesh


def load_depth_map(exr_file, height, width):
    """
    Load depth map from a .exr file.
    """
    file = OpenEXR.InputFile(exr_file)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_arr = array.array("f", file.channel("R", pixel_type))
    depth = np.array(depth_arr).reshape(height, width)

    # clip invalid depth
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0

    return depth


def save_point_cloud(pc, save_path):
    """
    Save a point cloud as a .ply file.
    """
    point_cloud = trimesh.PointCloud(pc)
    point_cloud.export(save_path)


def save_model_ckpt(state, experiment_dir, logger, new_best=False):
    """
    Save model checkpoint.
    """
    save_path = os.path.join(experiment_dir, "ckpts/ckpt_last.pth")
    logger.info("Saving model weights in %s"%save_path)
    torch.save(state, save_path)

    if new_best:
        save_path = os.path.join(experiment_dir, "ckpts/ckpt_best.pth")
        logger.info("New best model checkpoint. Saving model weights in %s"%save_path)
        torch.save(state, save_path)