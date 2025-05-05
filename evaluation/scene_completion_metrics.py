import numpy as np
import point_cloud_utils as pcu

from pykeops.numpy import LazyTensor


def collision(pred_pts, pred_meshes, background_mesh):
    """
    Collision metric for evaluating how badly predicted
    complete point cloud violates other predicted meshes.

    Args:
        pred_pts: predicted point set
        pred_meshes: list of predicted mesh objects composing a scene
        background_mesh: ground truth scene background mesh
    """
    collision_penalty = 0
    num_collisions = 0

    pc_size = 2048
    for idx in range(len(pred_meshes)):
        object_pc = pred_pts[idx * pc_size : (idx + 1) * pc_size].astype(np.float32)
        point_penalty = np.zeros((object_pc.shape[0],))

        v = np.array(background_mesh.vertices, dtype=np.float32)
        f = np.array(background_mesh.faces)
        b_sdf, fid, bc = pcu.signed_distance_to_mesh(object_pc, v, f)
        b_sdf *= -1

        point_penalty += np.abs(np.clip(b_sdf, a_min=-1e8, a_max=0.0))

        for mesh_idx, pred_mesh in enumerate(pred_meshes):
            if mesh_idx == idx:  # skip computing signed distance to itself
                continue

            v = np.array(pred_mesh.vertices, dtype=np.float32)
            f = np.array(pred_mesh.faces)
            sdf, fid, bc = pcu.signed_distance_to_mesh(object_pc, v, f)

            point_penalty += np.abs(np.clip(sdf, a_min=-1e8, a_max=0.0))
        
        collision_penalty += np.sum(point_penalty)
        num_collisions += np.sum(point_penalty > 0.0)

    collision_penalty /= pred_pts.shape[0]
    num_collisions /= pred_pts.shape[0]

    return collision_penalty, num_collisions


def chamfer_distance(pred_pts, gt_pts):
    """
    Chamfer distance between a ground truth point cloud and 
    predicted point cloud.

    Args:
        pred_pts: predicted point set
        gt_pts: ground truth point set
    Returns:
        cd: symmetric chamfer distance between ground truth 
            point set and predicted point set
    """
    x_i = LazyTensor(pred_pts[:, None, :])
    y_j = LazyTensor(gt_pts[None, :, :])
    D_ij = ((x_i - y_j)**2).sum(-1)**0.5

    d1 = D_ij.Kmin(1, axis=1, backend="CPU")
    d2 = D_ij.Kmin(1, axis=0, backend="CPU")

    cd = 0.5 * (np.mean(d1) + np.mean(d2))

    return cd


def one_sided_chamfer_distance(pred_pts, partial_pts):
    """
    One-sided chamfer distance between a partial point cloud and 
    predicted point cloud.

    Args:
        pred_pts: predicted point set
        partial_pts: partial point set
    Returns:
        cd: one sided chamfer distance between partial 
            point set and predicted point set
    """
    x_i = LazyTensor(pred_pts[:, None, :])
    y_j = LazyTensor(partial_pts[None, :, :])
    D_ij = ((x_i - y_j)**2).sum(-1)**0.5

    d2 = D_ij.Kmin(1, axis=0, backend="CPU")

    cd = np.mean(d2)

    return cd


def unidirectional_hausdorff_distance(pred_pts, partial_pts):
    """
    Unidirectional hausdorff distance between a partial point 
    cloud and predicted point cloud.

    Args:
        pred_pts: predicted point set
        partial_pts: partial point set
    Returns:
        cd: Unidirectional hausdorrf distance between partial 
            point set and predicted point set
    """
    x_i = LazyTensor(pred_pts[:, None, :])
    y_j = LazyTensor(partial_pts[None, :, :])
    D_ij = ((x_i - y_j)**2).sum(-1)**0.5

    uhd = np.max(D_ij.Kmin(1, axis=0, backend="CPU"))

    return uhd