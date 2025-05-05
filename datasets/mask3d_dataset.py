import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset

from utils.pc_utils import (
    normalize_point_cloud, sample_point_cloud
)


class Mask3DDataset(Dataset):
    def __init__(self, data_cfg):
        super().__init__()

        self.split = data_cfg.split
        self.num_renderings = data_cfg.num_renderings
        self.num_input_points = data_cfg.num_input_points
        self.num_gt_points = data_cfg.num_gt_points
        self.max_num_free_points = data_cfg.num_free_space_points
        self.max_num_occupied_points = data_cfg.num_occupied_space_points
        self.max_batch_size = data_cfg.max_batch_size

        self.data_root = data_cfg.dataset.data_root
        self.mask3d_data_root = data_cfg.dataset.mask3d_data_root

        # load ScanNet scene ids
        self.__load_scene_list()

        # load RGB colors for instances
        instance_color_mapper = self.__read_json(os.path.join(self.data_root, "instance_color_mapper.json"))
        instance_colors = [0] * len(instance_color_mapper)
        for key, value in instance_color_mapper.items():
            instance_colors[int(key)+1] = value
        self.instance_colors = np.array(instance_colors)

    def __load_scene_list(self):
        """
        Load scene ids for data loading.
        """
        scene_file_list = os.path.join(self.data_root, f"{self.split}_scene_ids.txt")
        with open(scene_file_list) as f:
            scene_ids = f.readlines()
            scene_id_list = [scene_id.strip() for scene_id in scene_ids]

        self.scene_id_list = []
        for scene_id in scene_id_list:
            self.scene_id_list.append(
                {
                    "scene_id": scene_id,
                    "partial_scan_id": 0,
                    "pred_mask_files": f"{self.mask3d_data_root}/{scene_id}_0.txt"
                }
            )
            self.scene_id_list.append(
                {
                    "scene_id": scene_id,
                    "partial_scan_id": 1,
                    "pred_mask_files": f"{self.mask3d_data_root}/{scene_id}_1.txt"
                }
            )

    def __read_json(self, filename):
        with open(filename, 'r') as infile:
            return json.load(infile)

    def __normalize(self, pc):
        """
        Generate transformations and apply them to the partial input.
        Additionally record transforms so they can be undone to align the input
        with the ground truth point cloud.
        """
        # normalize point cloud
        pc, norm_center, norm_scale = normalize_point_cloud(
            pc, 
            method="unit_cube",
            return_stats=True,
        )

        return pc, norm_center, norm_scale

    def __len__(self):
        return len(self.scene_id_list)

    def __get_scene(self, scene_dir, partial_id, pred_mask_files):
        """
        Load an entire scene in (used for validation and testing)
        """

        # load partial scene point cloud
        partial_scene_file = os.path.join(scene_dir, f"partials/partial_point_cloud_{str(partial_id)}.npz")
        partial_data = np.load(partial_scene_file, allow_pickle=True)
        partial_scene = []
        for key in partial_data["arr_0"].item():
            partial_object = partial_data["arr_0"].item()[key]
            partial_scene.append(partial_object)
        partial_scene = np.concatenate(partial_scene)
        
        partial = []
        partial_normals = []
        transforms = []
        classes = []
        scores = []

        # parse scene by Mask3D predicted object instances
        with open(pred_mask_files, "r") as f:
            d = f.readlines()

            for line in d:
                line_list = line.split(" ")
                pred_mask_file = os.path.join(self.mask3d_data_root, line_list[0])
                class_id = int(line_list[1]) - 1
                score = float(line_list[2])
                pred_mask = np.loadtxt(pred_mask_file)
                partial_object_pc = partial_scene[pred_mask == 1, :3]
                partial_object_normals = partial_scene[pred_mask == 1, 3:]

                partial_object_pc, partial_object_normals = sample_point_cloud(
                    partial_object_pc, self.num_input_points, partial_object_normals
                )
                pc, translation, scale = self.__normalize(partial_object_pc)

                partial.append(pc)
                partial_normals.append(partial_object_normals)
                transforms.append(np.concatenate([translation[None, :], np.array([[scale]])], axis=-1))    
                classes.append(class_id)
                scores.append(score)

        partial = torch.tensor(np.stack(partial, axis=0), dtype=torch.float32)
        partial_normals = torch.tensor(np.stack(partial_normals, axis=0), dtype=torch.float32)
        transforms = torch.tensor(np.stack(transforms, axis=0), dtype=torch.float32)
        classes = torch.tensor(np.stack(classes, axis=0), dtype=torch.int32)
        scores = torch.tensor(np.stack(scores, axis=0), dtype=torch.float32)

        return (
            partial, 
            partial_normals, 
            transforms, 
            classes,
            scores,
        )
    
    def __get_constraints(self, scene_dir, partial_id):
        """ 
        Load scene constraint point clouds
        """

        constraint_pc_file = os.path.join(
            scene_dir, 
            f"constraints/constraint_point_cloud_{str(partial_id)}.npz"
        )
        constraint_data = np.load(constraint_pc_file, allow_pickle=True)

        # get free space scene constraints
        free_space_pts = constraint_data["arr_0"].item()["free"]
        free_space_pts = sample_point_cloud(
            free_space_pts, self.max_num_free_points, clip=0.0001
        )
        free_space_pts = torch.tensor(free_space_pts, dtype=torch.float32)

        # get occupied space scene constraints
        occupied_space_pts = constraint_data["arr_0"].item()["occupied"]
        occupied_space_pts = sample_point_cloud(
            occupied_space_pts, self.max_num_occupied_points, clip=0.0001
        )
        occupied_space_pts = torch.tensor(occupied_space_pts, dtype=torch.float32)

        return free_space_pts, occupied_space_pts

    def __getitem__(self, idx):
        scene = self.scene_id_list[idx]
        scene_id = scene["scene_id"]
        partial_id = scene["partial_scan_id"]
        pred_mask_files = scene["pred_mask_files"]
        scene_dir = os.path.join(self.data_root, "scenes", scene_id)

        # load partial input and ground truth completion
        partial, partial_normals, transforms, classes, scores = \
            self.__get_scene(scene_dir, partial_id, pred_mask_files)

        # load scene constraints
        free_space_pts, occupied_space_pts = \
            self.__get_constraints(scene_dir, partial_id)

        return {
            "partial": partial, 
            "partial_normals": partial_normals,
            "free_space": free_space_pts,
            "occupied_space": occupied_space_pts,
            "transforms": transforms,
            "scene_id": scene_id,
            "partial_id": partial_id,
            "classes": classes,
            "scores": scores,
        }
