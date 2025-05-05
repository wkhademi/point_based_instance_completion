import torch.nn as nn
import torch.nn.functional as F

from models.archs.encoders.disco import PartialEncoder
from models.archs.decoders.seedformer import (
    ConstrainedSeedGenerator, CoarseToFineDecoder
)


class SCCNet(nn.Module):
    """
    Our Scene-Constrained Completion Network (SCCNet) used for 
    completing object instances within a scene while considering
    scene constraints.
    """
    def __init__(self, cfg):
        super().__init__()

        # partial point cloud encoder
        self.partial_encoder = PartialEncoder(**cfg.partial_encoder)

        # Constraint Aware and Robust Patch Seeds Generator
        self.seed_generator = ConstrainedSeedGenerator(**cfg.seed_generator)

        # point cloud completion decoder
        self.completion_decoder = CoarseToFineDecoder(**cfg.completion_decoder)
    
    def encode(self, partial_xyz_sets, partial_normal_sets):
        """
        Extract a set of local and global features from a partial point cloud.

        Args:
            partial_xyz_sets: list of partial points' xyz coordinates
                [[B, N1, 3], ..., [B, N5, 3]]
            partial_normal_sets: list of normals associated with partial_xyz_sets
                [[B, N1, 3], ..., [B, N5, 3]]
        Returns:
            local_xyz: xyz coordinates of local_features encoding partial shape
            local_features: per point feature corresponding to points local_xyz
            global_feature: global feature vector encoding partial shape
        """
        # coarsest sampling of point cloud describe local feature coordinates
        local_xyz = partial_xyz_sets[-1]

        # extract local and global features from partial point cloud
        local_features, global_feature = self.partial_encoder(
            partial_xyz_sets, partial_normal_sets
        )

        return local_xyz, local_features, global_feature

    def generate_patch_seeds(self, local_xyz, local_features, global_feature,
                             free_space_points, occupied_space_points, object_transforms):
        """
        Generate Patch Seeds of a complete shape from a set of partial shape
        features.

        Args:
            local_xyz: xyz coordinates of local_features encoding partial shape
            local_features: per point features corresponding to points local_xyz
            global_feature: global feature vector encoding partial shape
            free_space_points: point samples of free space in scene
            occupied_space_points: point samples of occupied space in scene
            object_transforms: per object normalization (scale + translation)
        Returns:
            seed_xyz: xyz coordinates of Patch Seeds encoding complete shape
            seed_features: per point feature corresponding to points seed_xyz
            object_center: predicted object center of complete shape ([B, 1, 3] or None)
        """
        seed_xyz, seed_features, object_center = self.seed_generator(
            local_xyz, 
            local_features, 
            global_feature,
            free_space_points,
            occupied_space_points,
            object_transforms,
        )

        return seed_xyz, seed_features, object_center

    def decode(self, partial_xyz, seed_xyz, seed_features):
        """
        Decode completion of partial shape in a coarse-to-fine scheme.

        Args:
            partial_xyz: xyz coordinates of partial shape
            seed_xyz: xyz coordinates of Patch Seeds encoding complete shape
            seed_features: per point feature corresponding to points seed_xyz
        Returns:
            completion_set: list of coarse-to-fine completions of partial shape
                [[B, N1, 3], ..., [B, N4, 3]]
            normals: predicted surface normals of highest resolution completion ([B, N4, 3] or None)
        """
        completion_set, normals = self.completion_decoder(
            partial_xyz, seed_xyz, seed_features
        )

        return completion_set, normals

    def forward(self, partial_xyz_sets, partial_normal_sets, free_space_points, 
                occupied_space_points, object_transforms):
        """
        Full forward pass for training a scene-constrained RobustSeedFormer.

        Args:
            partial_xyz_sets: list of partial points' xyz coordinates
                [[B, N1, 3], ..., [B, N5, 3]]
            partial_normal_sets: list of normals associated with partial_xyz_sets
                [[B, N1, 3], ..., [B, N5, 3]]
            free_space_points: point samples of free space in scene
            occupied_space_points: point samples of occupied space in scene
            object_transforms: per object normalization (scale + translation)
        Returns:
            completion_set: list of coarse-to-fine completions of partial shape
            normals: predicted surface normals of highest resolution completion
            object_center: predicted object center of complete shape
        """
        # extract local and global features from partial point cloud
        local_xyz, local_feats, global_feat = self.encode(
            partial_xyz_sets, partial_normal_sets
        )

        # generate Patch Seeds from partial shape information
        seed_xyz, seed_features, object_center = self.generate_patch_seeds(
            local_xyz, 
            local_feats, 
            global_feat, 
            free_space_points, 
            occupied_space_points,
            object_transforms,
        )

        # predict coarse-to-fine point cloud completion of partial shape
        partial_xyz = partial_xyz_sets[0]
        completion_set, normals = self.decode(
            partial_xyz, seed_xyz, seed_features
        )

        return completion_set, normals, object_center