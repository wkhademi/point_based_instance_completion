import os
import torch
import trimesh
import numpy as np

from torch.utils.data import Dataset

from utils.io_utils import load_mesh
from utils.pc_utils import (
    normalize_point_cloud, sample_point_cloud
)
from utils.misc_utils import (
    random_rotation, random_translation, random_scale, 
    read_json, make_M_from_tqs
)


class ScanWCFDataset(Dataset):
    def __init__(self, data_cfg):
        super().__init__()

        self.split = data_cfg.split
        self.augmentations = data_cfg.augmentations
        self.num_renderings = data_cfg.num_renderings
        self.num_input_points = data_cfg.num_input_points
        self.num_gt_points = data_cfg.num_gt_points
        self.max_num_free_points = data_cfg.num_free_space_points
        self.max_num_occupied_points = data_cfg.num_occupied_space_points
        self.max_batch_size = data_cfg.max_batch_size

        self.data_root = data_cfg.dataset.data_root
        self.shapenet_data_root = os.path.join(self.data_root, "shapenet_meshes")

        # load ScanNet scene ids
        self.__load_scene_list()

        # load RGB colors for instances
        instance_color_mapper = read_json(os.path.join(self.data_root, "instance_color_mapper.json"))
        instance_colors = [0] * len(instance_color_mapper)
        for key, value in instance_color_mapper.items():
            instance_colors[int(key)+1] = value
        self.instance_colors = np.array(instance_colors)

        if self.split == "val" or self.split == "test":
            self.augmentations = []

    def __load_scene_list(self):
        """
        Load scene ids for data loading.
        """
        scene_file_list = os.path.join(self.data_root, f"{self.split}_scene_ids.txt")
        with open(scene_file_list) as f:
            scene_ids = f.readlines()
            scene_id_list = [scene_id.strip() for scene_id in scene_ids]

        self.scene_id_list = []
        if self.split == "train":
            for scene_id in scene_id_list:
                for idx in range(self.num_renderings):
                    # find instance ids of objects in scene
                    scene_dir = os.path.join(self.data_root, "scenes", scene_id)
                    partial_scene_file = os.path.join(scene_dir, f"partials/partial_point_cloud_{str(idx)}.npz")
                    partial_data = np.load(partial_scene_file, allow_pickle=True)
                    for key in partial_data["arr_0"].item():
                        if int(key) == -1:  # ignore floor/walls for now
                            continue

                        self.scene_id_list.append(
                        {
                            "scene_id": scene_id,
                            "instance_id": key,
                            "partial_scan_id": idx,
                        }
                )
        else:
            for scene_id in scene_id_list:
                self.scene_id_list.append(
                    {
                        "scene_id": scene_id,
                        "instance_id": None,
                        "partial_scan_id": 0,
                    }
                )
                self.scene_id_list.append(
                    {
                        "scene_id": scene_id,
                        "instance_id": None,
                        "partial_scan_id": 1,
                    }
                )
    
    def __transform(self, pc):
        """
        Generate transformations and apply them to the partial input.
        Additionally record transforms so they can be undone to align the input
        with the ground truth point cloud.
        """
        transforms = {}

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

        return pc, transforms
    
    def __apply_transforms(self, pc, transforms):
        """
        Apply transformations to a point cloud.
        """
        
        if "scale" in transforms:
            pc *= transforms["scale"]

        if "translation" in transforms:
            pc += transforms["translation"]
        
        if "rotation" in transforms:
            pc = np.dot(pc, transforms["rotation"].T)

        return pc

    def __normalize(self, pc):
        # normalize point cloud
        pc, norm_center, norm_scale = normalize_point_cloud(
            pc, 
            method="unit_cube",
            return_stats=True,
        )

        return pc, norm_center, norm_scale

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

    def __load_gt_shapenet_model(self, model_annotation, partial_transform, augmentations=None):
        obj_id = model_annotation["cad_id"]
        cat_id = model_annotation["category_id"]
        cad_file = os.path.join(self.shapenet_data_root, cat_id, obj_id, "model.obj")
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

        # get ground truth centroid of object
        object_center = np.mean(object_mesh.vertices, axis=0)[None, :]

        # uniform sample mesh to get ground truth point cloud and normals
        object_pc, object_normals = self.__mesh_sample(object_mesh, self.num_gt_points)

        # transform GT point cloud to align it with the partial input
        object_pc = np.concatenate([object_pc, object_center], axis=0)
        if augmentations is not None:
            object_pc = self.__apply_transforms(object_pc, augmentations)
        object_pc = torch.tensor(object_pc, dtype=torch.float32)
        object_pc = (object_pc - partial_transform[..., :3]) / partial_transform[..., 3:]
        if augmentations is not None and "rotation" in augmentations:
            object_normals = np.dot(object_normals, augmentations["rotation"].T)

        object_center = object_pc[-1:, :]
        object_pc = object_pc[:-1, :]
        object_normals = torch.tensor(object_normals, dtype=torch.float32)

        return object_pc, object_normals, object_center

    def __len__(self):
        return len(self.scene_id_list)
    
    def __get_object(self, scene_dir, scene_id, instance_id, partial_id):
        """
        Load in a specific object from a scene (used for training)
        """

        # load in partial scan of scene
        partial_scene_file = os.path.join(
            scene_dir, 
            f"partials/partial_point_cloud_{str(partial_id)}.npz",
        )
        partial_data = np.load(partial_scene_file, allow_pickle=True)

        # get specific object from scene
        partial_object = partial_data["arr_0"].item()[instance_id]
        partial_object_pc = partial_object[:, :3]
        partial_object_normals = partial_object[:, 3:]

        # downsample partial input
        partial_object_pc, partial_object_normals = sample_point_cloud(
            partial_object_pc, self.num_input_points, partial_object_normals
        )

        # random augmentation
        partial_object_pc, augmentations = self.__transform(partial_object_pc)
        if "rotation" in augmentations:
            partial_object_normals = np.dot(
                partial_object_normals, augmentations["rotation"].T
            )

        # normalize partial input
        partial_object_pc, translation, scale = \
            self.__normalize(partial_object_pc)
        transform = np.concatenate(
            [translation[None, :], np.array([[scale]])], 
            axis=-1,
        )

        # convert to tensors
        partial = torch.tensor(partial_object_pc, dtype=torch.float32)
        partial_normals = torch.tensor(partial_object_normals, dtype=torch.float32)
        transform = torch.tensor(transform, dtype=torch.float32)
        
        # load in ground truth object
        gt_json_file = os.path.join(
            self.data_root, 
            f"json_files/{scene_id}.json",
        )
        scene_annotation_dict = read_json(gt_json_file)
        scene_annotations = scene_annotation_dict[scene_id]
        complete, complete_normals, complete_center = \
            self.__load_gt_shapenet_model(
                scene_annotations["instances"][instance_id], 
                transform,
                augmentations,
            )
        
        return (
            partial, 
            partial_normals,
            transform,
            augmentations,
            complete,
            complete_normals,
            complete_center,
        )

    def __get_scene(self, scene_dir, scene_id, partial_id):
        """
        Load an entire scene in (used for validation and testing)
        """

        # load partial scene point cloud
        partial_scene_file = os.path.join(scene_dir, f"partials/partial_point_cloud_{str(partial_id)}.npz")
        partial_data = np.load(partial_scene_file, allow_pickle=True)
        partial = []
        partial_normals = []
        transforms = []
        used_keys = []
        for key in partial_data["arr_0"].item():
            if int(key) == -1:  # ignore floor/walls for now
                continue

            used_keys.append(key)

            # normalize each partial object
            partial_object = partial_data["arr_0"].item()[key]
            partial_object_pc = partial_object[:, :3]
            partial_object_normals = partial_object[:, 3:]
            partial_object_pc, partial_object_normals = sample_point_cloud(
                partial_object_pc, self.num_input_points, partial_object_normals
            )
            pc, translation, scale = self.__normalize(partial_object_pc)

            partial.append(pc)
            partial_normals.append(partial_object_normals)
            transforms.append(np.concatenate([translation[None, :], np.array([[scale]])], axis=-1))    
        partial = torch.tensor(np.stack(partial, axis=0), dtype=torch.float32)
        partial_normals = torch.tensor(np.stack(partial_normals, axis=0), dtype=torch.float32)
        transforms = torch.tensor(np.stack(transforms, axis=0), dtype=torch.float32)

        # load ground truth scene point cloud
        gt_json_file = os.path.join(self.data_root, f"json_files/{scene_id}.json")
        scene_annotation_dict = read_json(gt_json_file)
        scene_annotations = scene_annotation_dict[scene_id]
        complete = []
        complete_normals = []
        complete_centers = []
        for idx, model_annotation in scene_annotations["instances"].items():
            if idx not in used_keys:
                continue

            object_pc, object_normals, object_center = \
                self.__load_gt_shapenet_model(
                    model_annotation, transforms[used_keys.index(idx)]
                )
            complete.append(object_pc)
            complete_normals.append(object_normals)
            complete_centers.append(object_center)
        complete = torch.stack(complete, dim=0)
        complete_normals = torch.stack(complete_normals, dim=0)
        complete_centers = torch.stack(complete_centers, dim=0)

        # trim off some object if too many (need to fix)
        partial = partial[:self.max_batch_size]
        partial_normals = partial_normals[:self.max_batch_size]
        transforms = transforms[:self.max_batch_size]
        complete = complete[:self.max_batch_size]
        complete_normals = complete_normals[:self.max_batch_size]
        complete_centers = complete_centers[:self.max_batch_size]

        return (
            partial, 
            partial_normals, 
            transforms, 
            None,
            complete, 
            complete_normals, 
            complete_centers,
        )
    
    def __get_constraints(self, scene_dir, partial_id, augmentations=None):
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
        if augmentations is not None:
            free_space_pts = self.__apply_transforms(free_space_pts, augmentations)
        free_space_pts = torch.tensor(free_space_pts, dtype=torch.float32)

        # get occupied space scene constraints
        occupied_space_pts = constraint_data["arr_0"].item()["occupied"]
        occupied_space_pts = sample_point_cloud(
            occupied_space_pts, self.max_num_occupied_points, clip=0.0001
        )
        if augmentations is not None:
            occupied_space_pts = self.__apply_transforms(occupied_space_pts, augmentations)
        occupied_space_pts = torch.tensor(occupied_space_pts, dtype=torch.float32)

        return free_space_pts, occupied_space_pts

    def __getitem__(self, idx):
        scene = self.scene_id_list[idx]
        scene_id = scene["scene_id"]
        instance_id = scene["instance_id"]
        partial_id = scene["partial_scan_id"]
        scene_dir = os.path.join(self.data_root, "scenes", scene_id)

        # load partial input and ground truth completion
        if self.split == "train":
            (partial, partial_normals, transforms, augmentations,
             complete, complete_normals, complete_centers) = \
                self.__get_object(
                    scene_dir, scene_id, instance_id, partial_id
                )
        else:
            (partial, partial_normals, transforms, augmentations,
             complete, complete_normals, complete_centers) = \
                self.__get_scene(scene_dir, scene_id, partial_id)

        # load scene constraints
        free_space_pts, occupied_space_pts = \
            self.__get_constraints(scene_dir, partial_id, augmentations)

        return {
            "partial": partial, 
            "partial_normals": partial_normals,
            "complete": complete, 
            "complete_normals": complete_normals,
            "object_center": complete_centers,
            "free_space": free_space_pts,
            "occupied_space": occupied_space_pts,
            "transforms": transforms,
            "scene_id": scene_id,
            "partial_id": partial_id,
        }
