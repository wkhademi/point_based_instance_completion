import os
import json
import torch
import random
import trimesh
import numpy as np
import point_cloud_utils as pcu

from pykeops.numpy import LazyTensor
from torch.utils.data import Dataset

from utils.io_utils import (
    load_depth_map, load_mesh
)
from utils.pc_utils import (
    normalize_point_cloud, sample_point_cloud, voxelize
)
from utils.misc_utils import (
    random_rotation, random_translation, random_scale
)


def num_view_mapper(num_models, ideal_num_models, max_num_views):
    if num_models >= ideal_num_models:
        return 1
    else:
        num_views = min(
            max_num_views,
            int(np.ceil(ideal_num_models / num_models)),
        )

        return num_views


class PretrainDataset(Dataset):
    def __init__(self, data_cfg):
        super().__init__()

        self.split = data_cfg.split
        self.augmentations = data_cfg.augmentations
        self.num_input_points = data_cfg.num_input_points
        self.input_downsample_points = data_cfg.input_downsample_points
        self.num_gt_points = data_cfg.num_gt_points
        self.gt_downsample_points = data_cfg.gt_downsample_points

        self.data_root = data_cfg.dataset.data_root
        model_list_path = data_cfg.dataset.model_list_path
        intrinsics_path = data_cfg.dataset.intrinsics_path

        num_views_per_category = data_cfg.num_views_per_category
        if self.split == "val" or self.split == "test":
            num_views_per_category = 1
            self.augmentations = []

        assert (num_views_per_category > 0 or num_views_per_category == -1), \
            "invalid choice for number of views"

        # load object models
        self.__load_model_list(model_list_path, num_views_per_category)

        # load intrinsics matrix
        self.__load_camera_intrisics(intrinsics_path)

    def __load_model_list(self, model_list_path, num_views_per_category):
        """
        Load object model names for data loading.
        """
        with open(model_list_path, "r") as f:
            data = json.load(f)

        ids = list(range(20))
        random.shuffle(ids)

        model_id_list = ["03325088", "04554684", "02880940", "02808440", "03797390",
                         "04330267", "03211117", "02818832", "02933112", "02773838",
                         "03337140", "02828884", "03207941", "02747177", "03636649",
                         "04468005", "02946921", "04256520", "02801938", "02871439",
                         "03001627", "02876657", "03593526", "04379243", "04460130",
                         "03642806", "02801938", "03938244", "03928116", "03085013",
                         "04004475", "03691459", "04074963", "03991062", "03046257",
                         "03761084", "03467517", "03790512", "02871439"]

        with open(os.path.join(self.data_root, "invalid_list.json"), "r") as f:
            invalid_data = json.load(f)

        invalid_list = []
        for category_id, model_ids in invalid_data.items():
            if category_id not in model_id_list:
                continue

            for model_id in model_ids:
                invalid_list.append(model_id) 

        self.model_list = []
        for category_id, model_ids in data.items():
            if category_id not in model_id_list:    
                continue

            if num_views_per_category == -1:
                num_views = num_view_mapper(len(model_ids), 2000, 20)
            else:
                num_views = num_views_per_category

            for idx, model_id in enumerate(model_ids):
                if model_id in invalid_list:
                    print(f"Skipped model id: {model_id}")
                    continue

                for i in range(num_views):
                    view_id = ids[i]
                    self.model_list.append((category_id, model_id, view_id))

    def __load_camera_intrisics(self, intrinsics_path):
        """
        Load intrisic matrix of camera.
        """
        self.K = np.loadtxt(intrinsics_path)
        self.inv_K = np.linalg.inv(self.K)
        self.inv_K[2, 2] = -1

    def __load_camera_pose(self, pose_path):
        """
        Load the camera pose at which depth map was rendered.
        """
        pose = np.loadtxt(pose_path)

        return pose

    def __read_exr(self, exr_file):
        """
        Read 16 bit depth map from .exr file.
        """
        H = int(self.K[1, 2] * 2)
        W = int(self.K[0, 2] * 2)
        depth = load_depth_map(exr_file, H, W)
        depth[depth > 2.] = 0  # clip anything outside unit cube

        return depth

    def __backproject(self, depth, pose):
        """
        Backproject depth map into 3D in world coordinate system.
        """
        depth = np.flipud(depth)
        y, x = np.where(depth > 0)

        # image coords --> camera coords
        hom_image_coords = np.stack([x, y, np.ones_like(x)], axis=0)
        cam_coords = np.dot(self.inv_K, hom_image_coords * depth[y, x])

        # camera coords --> world coords
        hom_cam_coords = np.concatenate(
            [cam_coords, np.ones((1, cam_coords.shape[1]))], axis=0
        )
        world_coords = np.dot(pose, hom_cam_coords).T[:, :3]

        # get view direction
        cam_origin = pose[:3, 3:4].T
        view_dirs = world_coords - cam_origin

        # estimate normals
        _, normals = pcu.estimate_point_cloud_normals_knn(
            world_coords, 
            num_neighbors=16, 
            view_directions=-1 * view_dirs,
        )

        return world_coords, normals

    def __transform(self, pc):
        """
        Generate transformations and apply them to the partial input.
        Additionally record transforms so they can be undone to align the input
        with the ground truth point cloud.
        """
        transforms = {}

        # scale between mesh and partial point cloud
        transforms["mesh_scale"] = 0.65

        # apply random scale
        if "scale" in self.augmentations:
            scale = random_scale()
            pc *= scale
            transforms["scale"] = scale

        # apply random translation
        if "translation" in self.augmentations:
            t = random_translation()
            pc += t
            transforms["translation"] = t

        # apply random rotation
        if "rotation" in self.augmentations:
            R = random_rotation()
            pc = np.dot(pc, R.T)
            transforms["rotation"] = R 

        # normalize point cloud
        pc, norm_center, norm_scale = normalize_point_cloud(
            pc, 
            method="unit_cube",
            return_stats=True,
        )
        transforms["norm_center"] = norm_center
        transforms["norm_scale"] = norm_scale

        return pc, transforms

    def __apply_transforms(self, pc, transforms):
        """
        Apply transformations to a point cloud.
        """
        pc *= transforms["mesh_scale"]
        
        if "scale" in transforms:
            pc *= transforms["scale"]

        if "translation" in transforms:
            pc += transforms["translation"]
        
        if "rotation" in transforms:
            pc = np.dot(pc, transforms["rotation"].T)

        pc -= transforms["norm_center"]
        pc /= transforms["norm_scale"]

        return pc

    def __grid_subsample(self, pc, normals, num_points):
        """
        Generate set of downsampled point clouds with the number of points
        defined in the list num_points using grid subsampling.
        """
        torch_pcs = [torch.tensor(pc, dtype=torch.float32)]

        if normals is not None:
            torch_normals = [torch.tensor(normals, dtype=torch.float32)]

        for num_point in num_points:
            if len(pc) != num_point:
                x_i = LazyTensor(pc[:, None, :])
                y_j = LazyTensor(pc[None, :, :])
                D_ij = ((x_i - y_j)**2).sum(-1)**0.5
                dist = D_ij.Kmin(2, 1, backend="CPU")[:, 1]
                voxel_size = 2 * np.mean(dist)

                idx = voxelize(pc, voxel_size=voxel_size)
                pc = pc[idx, :]
                if normals is not None:
                    normals = normals[idx, :]
                pc, normals = sample_point_cloud(pc, num_point, normals)

            torch_pcs.append(torch.tensor(pc, dtype=torch.float32))

            if normals is not None:
                torch_normals.append(torch.tensor(normals, dtype=torch.float32))

        if normals is not None:
            return torch_pcs, torch_normals
        else:
            return torch_pcs

    def __mesh_sample(self, mesh, num_samples):
        samples, fid = mesh.sample(num_samples, return_index=True)

        # point cloud
        pc = np.array(samples, dtype=np.float32)

        # compute normals from mesh
        bary = trimesh.triangles.points_to_barycentric(
            triangles=mesh.triangles[fid], points=samples
        )
        normals = np.array(
            trimesh.unitize((
                mesh.vertex_normals[mesh.faces[fid]] *
                    trimesh.unitize(bary).reshape((-1, 3, 1))
            ).sum(axis=1)),
        )

        return pc, normals

    def __len__(self):
        return len(self.model_list)

    def __getitem__(self, idx):
        category_id, model_id, view_id = self.model_list[idx]
        view_id = random.randint(0, 19)

        model_dir = os.path.join(self.data_root, category_id, model_id)
        depth_file = os.path.join(model_dir, f"depth/{str(view_id)}.exr")
        pose_file = os.path.join(model_dir, f"pose/{str(view_id)}.txt")
        obj_file = os.path.join(model_dir, "model.obj")

        # load partial point cloud and estimated normals
        depth = self.__read_exr(depth_file)
        pose = self.__load_camera_pose(pose_file)
        partial_pc, partial_normals = self.__backproject(depth, pose)
        partial_pc, partial_normals = sample_point_cloud(
            partial_pc, self.num_input_points, partial_normals
        )

        # apply transforms to partial point cloud
        partial_pc, transforms = self.__transform(partial_pc)
        if "rotation" in transforms:
            partial_normals = np.dot(partial_normals, transforms["rotation"].T)

        # create downsampled versions of partial point cloud
        partial, partial_normals = self.__grid_subsample(
            partial_pc, partial_normals, num_points=self.input_downsample_points
        )   

        # load GT mesh and sample GT point cloud from it
        mesh = load_mesh(obj_file)
        object_center = np.mean(mesh.vertices, axis=0)[None, :]
        complete_pc, complete_normals = self.__mesh_sample(mesh, self.num_gt_points)
        complete_pc = np.concatenate([complete_pc, object_center], axis=0)

        # transform GT point cloud to align it with the partial input
        complete_pc = self.__apply_transforms(complete_pc, transforms)
        object_center = complete_pc[-1:, :]
        complete_pc = complete_pc[:-1, :]
        object_center = torch.tensor(object_center, dtype=torch.float32)
        complete = torch.tensor(complete_pc, dtype=torch.float32)
        if "rotation" in transforms:
            complete_normals = np.dot(complete_normals, transforms["rotation"].T)
        complete_normals = torch.tensor(complete_normals, dtype=torch.float32)

        return {
            "partial": partial, 
            "partial_normals": partial_normals,
            "complete": complete, 
            "complete_normals": complete_normals,
            "object_center": object_center,
            "transforms": transforms,
            "category_id": category_id,
            "model_id": model_id,
            "view_id": view_id,
        }
