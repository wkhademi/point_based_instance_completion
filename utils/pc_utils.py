import torch
import random
import numpy as np

from pykeops.torch import LazyTensor

from pointnet2_ops.pointnet2_utils import fps
from utils.misc_utils import fnv_hash_vec, ravel_hash_vec


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode='random'):
    '''
    Voxelization of the input coordinates
    Parameters:
        coord: input coordinates (N x D)
        voxel_size: Size of the voxels
        hash_type: Type of the hashing function, can be chosen from 'ravel' and 'fnv'
        mode: 'random', 'deterministic' or 'multiple' mode. In training mode one selects a random point within the voxel as the representation of the voxel.
              In deterministic model right now one always uses the first point. Usually random mode is preferred for training. In 'multiple' mode, we will return
              multiple sets of indices, so that each point will be covered in at least one of these sets
    Returns:
        idx_unique: the indices of the points so that there is at most one point for each voxel
    '''
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 'deterministic':
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + torch.randint(count.max(), (count.size,)).numpy() % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.zeros((count.size,), dtype=np.int32)
        idx_unique = idx_sort[idx_select]
        return idx_unique
    elif mode == 'multiple':  # mode is 'multiple'
        idx_data = []
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
        return idx_data
    else:  # mode == 'random'
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + torch.randint(count.max(), (count.size,)).numpy() % count
        idx_unique = idx_sort[idx_select]
        return idx_unique


def knn(pts1, pts2, k):
    """
    Compute k-nearest neighbors in one point set to another.

    Args:
        pts1: query point set, [B, S, C]
        pts2: other point set, [B, N, C]
        k: number of nearest neighbors to sample
    Return:
        indices_i: indices of points' nearest neighbors, [B, S, k]
    """
    with torch.no_grad():
        B, S, C = pts1.shape
        _, N, _ = pts2.shape

        x_i = LazyTensor(pts1.view(B, S, 1, C))
        y_j = LazyTensor(pts2.view(B, 1, N, C))

        D_ij = ((x_i - y_j)**2).sum(-1)**0.5
        indices_i = D_ij.argKmin(k, dim=2)

    return indices_i.int()


def normalize_point_cloud(pc, method="unit_sphere", return_stats=False):
    """
    Normalize a point cloud.

    Args:
        pc: point cloud to be normalized.
        method: whether to normalize point cloud to unit sphere or unit cube
                [unit_sphere | unit_cube].
    Returns:
        pc: normalized version of the input point cloud.
    """
    if method == "unit_cube":
        max_bounds = np.max(pc, axis=0)
        min_bounds = np.min(pc, axis=0)
        center = (max_bounds + min_bounds) / 2.
        scale = (max_bounds - min_bounds).max()
        pc -= center
        pc /= scale
    elif method == "unit_sphere":
        center = np.mean(pc, axis=0)
        pc -= center
        scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc /= scale

    if return_stats:
        return pc, center, scale
    else:
        return pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly shift the individual points of a point cloud.

    Args:
        pc: point cloud to add noise to.
        sigma: standard deviation of noise.
        clip: clip any translations that exceed this value.
    Returns:
        pc: numpy array (N, 3)
    """
    N, C = pc.shape
    shift = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    pc = pc + shift

    return pc


def downsample_point_cloud(pc, num_samples, normals=None):
    """
    Downsample point cloud by random choice.

    Args:
        pc: point cloud to downsample.
        normals: surface normals associated with pc
        num_samples: desired number of samples to have in downsampled point cloud.
    Returns:
        pc: input point cloud downsampled to have num_samples points
        normals: input normals downsampled with pc
    """
    indices = random.sample(list(range(len(pc))), k=num_samples)
    pc = pc[indices]

    if normals is not None:
        normals = normals[indices]

    return pc, normals


def upsample_point_cloud(pc, num_samples, normals=None, clip=0.001):
    """
    Upsample point cloud by random choice.

    Args:
        pc: point cloud to upsample.
        normals: surface normals associated with pc
        num_samples: desired number of samples to have in upsampled point cloud.
    Returns:
        pc: input point cloud upsampled to have num_samples points.
        normals: input normals upsampled with pc
    """
    curr = len(pc)
    need = num_samples - curr

    while curr <= need:
        duplicated_pts = jitter_point_cloud(pc, sigma=0.0005, clip=clip)
        pc = np.concatenate([pc, duplicated_pts], axis=0)

        if normals is not None:
            normals = np.concatenate([normals, normals], axis=0)

        need -= curr
        curr *= 2

    choice = np.random.permutation(need)
    pc = np.concatenate([pc, pc[choice]], axis=0)

    if normals is not None:
        normals = np.concatenate([normals, normals[choice]], axis=0)
    
    return pc, normals


def sample_point_cloud(pc, num_samples, normals=None, clip=0.001):
    """
    Sample num_samples points from a point cloud. If num_samples is larger than
    the number of points in the point cloud, then points are randomly duplicated
    and their positions are slightly jittered.

    Args:
        pc: point cloud.
        normals: surface normals associated with pc
        num_samples: number of points to sample from the input point cloud.
    Returns:
        pc: input point cloud resampled to have num_samples point.
        normals: input normals resampled with pc
    """
    if len(pc) > num_samples:
        pc, normals = downsample_point_cloud(pc, num_samples, normals)
    elif len(pc) < num_samples:
        pc, normals = upsample_point_cloud(pc, num_samples, normals, clip=clip)
    else:
        pc = pc
        normals = normals

    if normals is not None:
        return pc, normals
    else:
        return pc


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def batch_downsample(pc, n_samples, normals=None):
    """
    Perform furthest point sampling on a batch of point clouds. This
    function repeatedly downsamples the point cloud to each size in
    list n_samples. Optionally downsample the surface normals associated
    with the point set.

    Args:
        pc: batch of point clouds [B, N, 3]
        n_samples: list of values to downsample pc to
        normals (optional): surface normals of pc [B, N, 3]
    Returns:
        downsampled_pc: list of batched point clouds containing each 
                        downsampled point set size in n_samples
        downsampled_normals: corresponding downsampled normals
    """
    downsampled_pc = []
    if normals is not None:
        downsampled_normals = []

    for n_sample in n_samples:
        idx = fps(pc, n_sample)
        pc = index_points(pc, idx)
        downsampled_pc.append(pc)

        if normals is not None:
            normals = index_points(normals, idx)
            downsampled_normals.append(normals)

    if normals is not None:
        return downsampled_pc, downsampled_normals
    else:
        return downsampled_pc