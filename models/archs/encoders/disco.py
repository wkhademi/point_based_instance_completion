import torch
import torch.nn as nn
import torch.nn.functional as F

from models.archs.mlp import MLP
from models.archs.pointconv import (
    BottleneckPointConv, PointConv
)

from utils.pc_utils import knn


class PCFE(nn.Module):
    """
    PointConv Feature Extraction module.
    """
    def __init__(self, point_dim, channel_in, channel_out, k=16,
                 bottleneck=4, c_mid=4, use_vi=False):
        super().__init__()

        self.k = k

        self.interpolation_layer = (
            PointConv(
                k=k,
                point_dim=point_dim,
                in_channel=channel_in,
                out_channel=channel_out,
                c_mid=c_mid,
                use_vi=use_vi,
            )
        )

        self.pointconv_resblocks = nn.ModuleList()
        for _ in range(1):
            self.pointconv_resblocks.append(
                BottleneckPointConv(
                    k=k,
                    point_dim=point_dim,
                    in_channel=channel_out,
                    out_channel=channel_out,
                    bottleneck=bottleneck,
                    c_mid=c_mid,
                    use_vi=use_vi,
                )
            )

    def forward(self, xyz, features, new_xyz, normals, new_normals):
        """
        Args:
            xyz: input points' 3D location [B, N, 3]
            features: input points' feature [B, N, C]
            new_xyz: output points' 3D location [B, S, 3]
            normals: surface normals associated with xyz [B, N, 3]
            new_normals: surface normals associated with new_xyz [B, S, 3]
        Returns:
            new_features: output points' feature [B, S, D]
        """
        interpolated_features = self.interpolation_layer(
            xyz, features, new_xyz, normals, new_normals
        )

        nn_idx = knn(new_xyz, new_xyz, k=self.k)

        new_features = interpolated_features
        for pointconv_resblock in self.pointconv_resblocks:
            new_features = pointconv_resblock(
                new_xyz, new_features, new_normals, nn_idx
            )

        return new_features


class PartialEncoder(nn.Module):
    """
    Our point cloud encoder for extracting features from a partial object.
     
    Based on partial encoder from Diverse Shape Completion via Style
    Modulated Generative Adversarial Networks [Khademi and Li, NeurIPS 2023].
    """
    def __init__(self, base_feature_dim=16, global_feature_dim=512,
                 local_feature_dims=[32, 64, 128, 256], k=16, bottleneck=4,
                 c_mid=4, use_vi=False, include_normals=False):
        super().__init__()

        in_dim = 3

        self.include_normals = include_normals
        if include_normals:
            in_dim += 3

        self.pointwise_encode = True if base_feature_dim != 0 else False

        # point wise encoder
        if self.pointwise_encode:
            self.pw_mlp = MLP(
                in_dim=in_dim,
                out_dim=base_feature_dim,
                hidden_dims=[base_feature_dim, base_feature_dim],
            )

        # local feature extractor
        self.fe_layers = nn.ModuleList()
        channel_in = base_feature_dim if self.pointwise_encode else in_dim
        for channel_out in local_feature_dims:
            self.fe_layers.append(
                PCFE(
                    3,
                    channel_in,
                    channel_out,
                    k,
                    bottleneck,
                    c_mid,
                    use_vi,
                )
            )
            channel_in = channel_out

        # global feature extractor
        self.global_mlp = MLP(
            in_dim=channel_out + 3,
            out_dim=global_feature_dim,
            hidden_dims=[global_feature_dim],
            final_norm=False,
            final_act=False,
        )

    def forward(self, xyz_sets, normal_sets):
        """
        Args:
            xyz_sets: list of points' xyz location [[B, N1, 3], ..., [B, N5, 3]]
            normal_sets: list of normals associated with xyz_sets
                [[B, N1, 3], ..., [B, N5, 3]]
        Returns:
            local_features: features of downsampled partial shape [B, N5, C]
            global_feature: global feature descriptor of partial shape [B, 1, D]
        """
        xyz = xyz_sets[0]
        normals = normal_sets[0]

        if self.include_normals:
            input = torch.cat([xyz, normals], dim=-1)
        else:
            input = xyz

        if self.pointwise_encode:
            features = self.pw_mlp(input)  # point wise encoding
        else:
            features = input

        # extract local shape information from partial input
        for idx, fe_layer in enumerate(self.fe_layers):
            features = fe_layer(
                xyz_sets[idx], 
                features, 
                xyz_sets[idx+1], 
                normal_sets[idx], 
                normal_sets[idx+1]
            )
        local_features = features

        # extract global feature vector
        features = torch.cat([xyz_sets[-1], features], dim=-1)
        global_feature = self.global_mlp(features)
        global_feature = torch.max(global_feature, dim=1, keepdim=True)[0]

        return local_features, global_feature
