import torch.nn as nn
import torch.nn.functional as F

from models.archs.encoders.disco import PartialEncoder
from models.archs.decoders.seedformer import (
    PatchSeedGenerator, RobustSeedGenerator, CoarseToFineDecoder
)


class SeedFormer(nn.Module):
    """
    Our object-level completion model used in our scene completion network.

    Based on SeedFormer model from SeedFormer: Patch Seeds based Point Cloud 
    Completion with Upsample Transformer [Zhou et al., ECCV 2022].
    """
    def __init__(self, cfg):
        super().__init__()

        # partial point cloud encoder
        self.partial_encoder = PartialEncoder(**cfg.partial_encoder)

        # Patch Seeds generator
        if cfg.model_name == "SeedFormer":
            # local attention based seed generator from SeedFormer
            self.seed_generator = PatchSeedGenerator(**cfg.seed_generator)
        elif cfg.model_name == "RobustSeedFormer":
            # our global attention based seed generator
            self.seed_generator = RobustSeedGenerator(**cfg.seed_generator)

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

    def generate_patch_seeds(self, local_xyz, local_features, global_feature):
        """
        Generate Patch Seeds of a complete shape from a set of partial shape
        features.

        Args:
            local_xyz: xyz coordinates of local_features encoding partial shape
            local_features: per point features corresponding to points local_xyz
            global_feature: global feature vector encoding partial shape
        Returns:
            seed_xyz: xyz coordinates of Patch Seeds encoding complete shape
            seed_features: per point feature corresponding to points seed_xyz
            object_center: predicted object center of complete shape ([B, 1, 3] or None)
        """
        seed_xyz, seed_features, object_center = self.seed_generator(
            local_xyz, local_features, global_feature
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

    def forward(self, partial_xyz_sets, partial_normal_sets):
        """
        Full forward pass for object-level completion model.

        Args:
            partial_xyz_sets: list of partial points' xyz coordinates
                [[B, N1, 3], ..., [B, N5, 3]]
            partial_normal_sets: list of normals associated with partial_xyz_sets
                [[B, N1, 3], ..., [B, N5, 3]]
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
            local_xyz, local_feats, global_feat
        )

        # predict coarse-to-fine point cloud completion of partial shape
        partial_xyz = partial_xyz_sets[0]
        completion_set, normals = self.decode(
            partial_xyz, seed_xyz, seed_features
        )

        return completion_set, normals, object_center