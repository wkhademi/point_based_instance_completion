"""
This file contains a modified version of the Patch Seed generation and
upsampling layers from SeedFormer: Patch Seeds based Point Cloud Completion
with Upsample Transformer [Zhou et al., ECCV 2022].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_utils import fps

from utils.pc_utils import index_points
from models.archs.mlp import MLP, MLP_Res
from models.archs.layer_utils import Upsample
from models.archs.pointconv import BottleneckInterpPointConv
from models.archs.transformer import (
    UpsampleTransformer, TransformerBlock, CrossTransformerBlock, UpsampleTransformerBlock
)


class NormalEstimator(nn.Module):
    """
    Normal Estimation layer.
    """
    def __init__(self, in_dim, out_dim, interpolation_params,
                 attn_dim=64, k=20):
        super().__init__()

        self.k = k

        # Patch Seed Interpolation layer
        self.seed_interpolation = BottleneckInterpPointConv(
            point_dim=3,
            in_channel=in_dim,
            out_channel=out_dim,
            **interpolation_params,
        )

        # Coordinate MLP
        self.mlp1 = MLP(
            in_dim=3,
            out_dim=out_dim,
            hidden_dims=[out_dim // 2],
            final_norm=False,
            final_act=False,
        )
        self.mlp2 = MLP(
            in_dim=3 * out_dim,
            out_dim=out_dim,
            hidden_dims=[2 * out_dim],
            final_norm=False,
            final_act=False,
        )

        # Transformer layers
        self.attn = UpsampleTransformer(
            in_dim,
            out_dim,
            dim=attn_dim,
            n_knn=k,
            use_upfeat=True,
            up_factor=None,
        )

        # Normal prediction head
        self.normal_mlp = MLP(
            in_dim=out_dim,
            out_dim=3,
            hidden_dims=[out_dim // 2],
            final_norm=False,
            final_act=False,
        )

    def forward(self, xyz, prev_features, seed_xyz, seed_features):
        """
        Args:
            xyz: a set of xyz coordinates [B, N, 3]
            prev_features: features output by previous Upsample Transformer
                layer [B, N, C]
            seed_xyz: Patch Seeds coordinates [B, N1, 3]
            seed_features: Patch Seeds features [B, N1, C]
        Returns:
            normals: estimated surface normals of points xyz [B, N, 3]
        """
        # interpolate features from Patch Seeds
        interpolated_features = self.seed_interpolation(
            seed_xyz, seed_features, xyz
        )

        # extract features from xyz coordinates
        features = self.mlp1(xyz)
        features = self.mlp2(
            torch.cat(
                [
                    features,
                    torch.max(
                        features,
                        dim=1,
                        keepdim=True
                    )[0].expand_as(features),
                    interpolated_features,
                ],
                dim=-1,
            )
        )

        # local attention
        features = self.attn(
            pos=xyz,
            key=prev_features if prev_features is not None else features,
            query=features,
            upfeat=interpolated_features,
        )

        # normal prediction
        normals = self.normal_mlp(features)

        return normals


class Upsampler(nn.Module):
    """
    Upsampling layer.
    """
    def __init__(self, i, in_dim, out_dim, interpolation_params,
                 attn_dim=64, up_factor=2, k=20, radius=1):
        super().__init__()

        self.i = i
        self.k = k
        self.radius = radius
        self.up_factor = up_factor

        # Patch Seed Interpolation layer
        self.seed_interpolation = BottleneckInterpPointConv(
            point_dim=3,
            in_channel=in_dim,
            out_channel=out_dim,
            **interpolation_params,
        )

        # Coordinate MLP
        self.mlp1 = MLP(
            in_dim=3,
            out_dim=out_dim,
            hidden_dims=[out_dim // 2],
            final_norm=False,
            final_act=False,
        )
        self.mlp2 = MLP(
            in_dim=3 * out_dim,
            out_dim=out_dim,
            hidden_dims=[2 * out_dim],
            final_norm=False,
            final_act=False,
        )

        # Transformer layers
        self.attn = UpsampleTransformer(
            in_dim,
            out_dim,
            dim=attn_dim,
            n_knn=k,
            use_upfeat=True,
            up_factor=None,
        )
        
        self.upsample_attn = UpsampleTransformer(
            in_dim,
            out_dim,
            dim=attn_dim,
            n_knn=k,
            use_upfeat=True,
            up_factor=up_factor,
        )

        # nearest neighbor upsample operation
        self.upsample = Upsample(scale_factor=up_factor)

        # Coordinate offset prediction (up_factor offsets predicted per point)
        self.feature_mlp = MLP_Res(
            in_dim=2 * out_dim,
            hidden_dim=out_dim,
            out_dim=out_dim,
            final_act=False,
        )
        self.offset_mlp = MLP(
            in_dim=out_dim,
            out_dim=3,
            hidden_dims=[out_dim // 2],
            final_norm=False,
            final_act=False,
        )

    def forward(self, xyz, prev_features, seed_xyz, seed_features):
        """
        Args:
            xyz: a set of xyz coordinates [B, N, 3]
            prev_features: features output by previous Upsample Transformer
                layer [B, N, C]
            seed_xyz: Patch Seeds coordinates [B, N1, 3]
            seed_features: Patch Seeds features [B, N1, C]
        Returns:
            upsampled_xyz: upsampled set of xyz coordinates [B, N*up_factor, 3]
            upsampled_features: upsampled set of features [B, N*up_factor, C]
        """
        # interpolate features from Patch Seeds
        interpolated_features = self.seed_interpolation(
            seed_xyz, seed_features, xyz
        )

        # extract features from xyz coordinates
        features = self.mlp1(xyz)
        features = self.mlp2(
            torch.cat(
                [
                    features,
                    torch.max(
                        features,
                        dim=1,
                        keepdim=True
                    )[0].expand_as(features),
                    interpolated_features,
                ],
                dim=-1,
            )
        )

        # attention based upsampling/splitting of features
        features = self.attn(
            pos=xyz,
            key=prev_features if prev_features is not None else features,
            query=features,
            upfeat=interpolated_features,
        )
        upsampled_features = self.upsample_attn(
            pos=xyz,
            key=prev_features if prev_features is not None else features,
            query=features,
            upfeat=interpolated_features,
        )

        # nearest neighbor upsample of current layers coordinates and features
        up_features = self.upsample(features)
        xyz = self.upsample(xyz)

        # produce final upsampled features
        upsampled_features = self.feature_mlp(
            torch.cat([upsampled_features, up_features], dim=-1)
        )

        # offset prediction
        offsets = self.offset_mlp(F.leaky_relu(upsampled_features, 0.2))
        offsets = F.tanh(offsets) / self.radius**self.i

        # upsampled xyz coordinates
        upsampled_xyz = xyz + offsets
        upsampled_xyz = upsampled_xyz.contiguous()

        return upsampled_xyz, upsampled_features


class RobustUpsampler(nn.Module):
    def __init__(self, i, in_dim, out_dim, interpolation_params, attn_dim=64,
                 up_factor=2, k=20, radius=1, num_heads=8, num_freqs=64):
        super().__init__()
        
        self.i = i
        self.k = k
        self.radius = radius
        self.up_factor = up_factor

        # Patch Seed Interpolation layer
        self.seed_interpolation = BottleneckInterpPointConv(
            point_dim=3,
            in_channel=in_dim,
            out_channel=out_dim,
            **interpolation_params,
        )

        # Coordinate MLP
        self.mlp1 = MLP(
            in_dim=3,
            out_dim=out_dim,
            hidden_dims=[out_dim // 2],
            final_norm=False,
            final_act=False,
        )
        self.mlp2 = MLP(
            in_dim=3 * out_dim,
            out_dim=out_dim,
            hidden_dims=[2 * out_dim],
            final_norm=False,
            final_act=False,
        )

        # positional encoding
        self.pos_embed= MLP(
            in_dim=3,
            out_dim=out_dim,
            hidden_dims=[in_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # transformer blocks with global self attention
        self.transformer_blocks = nn.ModuleList()
        for i in range(3):
            self.transformer_blocks.append(
                TransformerBlock(
                    out_dim,
                    num_heads,
                    attn_drop=0.0,
                    res_drop=0.0,
                )
            )

        # Transformer layers
        self.attn = UpsampleTransformer(
            out_dim,
            out_dim,
            dim=attn_dim,
            n_knn=k,
            use_upfeat=True,
            up_factor=None,
        )
        
        self.upsample_attn = UpsampleTransformer(
            out_dim,
            out_dim,
            dim=attn_dim,
            n_knn=k,
            use_upfeat=True,
            up_factor=up_factor,
        )

        # nearest neighbor upsample operation
        self.upsample = Upsample(scale_factor=up_factor)

        # Coordinate offset prediction (up_factor offsets predicted per point)
        self.feature_mlp = MLP_Res(
            in_dim=2 * out_dim,
            hidden_dim=out_dim,
            out_dim=out_dim,
            final_act=False,
        )
        self.offset_mlp = MLP(
            in_dim=out_dim,
            out_dim=3,
            hidden_dims=[out_dim // 2],
            final_norm=False,
            final_act=False,
        )

    def forward(self, xyz, prev_features, seed_xyz, seed_features):
        """
        Args:
            xyz: a set of xyz coordinates [B, N, 3]
            prev_features: features output by previous Upsample Transformer
                layer [B, N, C]
            seed_xyz: Patch Seeds coordinates [B, N1, 3]
            seed_features: Patch Seeds features [B, N1, C]
        Returns:
            upsampled_xyz: upsampled set of xyz coordinates [B, N*up_factor, 3]
            upsampled_features: upsampled set of features [B, N*up_factor, C]
        """
        # interpolate features from Patch Seeds
        interpolated_features = self.seed_interpolation(
            seed_xyz, seed_features, xyz
        )

        # extract features from xyz coordinates [B, N, out_dim]
        features = self.mlp1(xyz)
        feat_embed = self.mlp2(
            torch.cat(
                [
                    features,
                    torch.max(
                        features,
                        dim=1,
                        keepdim=True
                    )[0].expand_as(features),
                    interpolated_features,
                ],
                dim=-1,
            )
        )

        # positional encoding [B, N, out_dim]
        pos_embed = self.pos_embed(xyz)

        # apply transformer blocks with global attention
        embed = pos_embed + feat_embed
        for blk in self.transformer_blocks:
            embed = blk(embed)
        features = embed

        # local attention based upsampling/splitting of features
        features = self.attn(
            pos=xyz,
            key=prev_features if prev_features is not None else features,
            query=features,
            upfeat=interpolated_features,
        )
        upsampled_features = self.upsample_attn(
            pos=xyz,
            key=prev_features if prev_features is not None else features,
            query=features,
            upfeat=interpolated_features,
        )

        # nearest neighbor upsample of current layers coordinates and features
        up_features = self.upsample(features)
        xyz = self.upsample(xyz)

        # produce final upsampled features
        upsampled_features = self.feature_mlp(
            torch.cat([upsampled_features, up_features], dim=-1)
        )

        # offset prediction
        offsets = self.offset_mlp(F.leaky_relu(upsampled_features, 0.2))

        # upsampled xyz coordinates
        upsampled_xyz = xyz + offsets
        upsampled_xyz = upsampled_xyz.contiguous()

        return upsampled_xyz, upsampled_features


class CoarseToFineDecoder(nn.Module):
    """
    Coarse to Fine Decoder for producing dense completions of partial point
    clouds.
    """

    def __init__(self, interpolation_params, up_factors=[1, 4, 4], dim=128, 
                 hidden_dim=64, k=20, radius=1, robust_upsampler=False,
                 predict_normals=False):
        super().__init__()

        if robust_upsampler:
            upsampler = RobustUpsampler
        else:
            upsampler = Upsampler

        # upsample point cloud
        self.upsample_layers = nn.ModuleList()
        for i, up_factor in enumerate(up_factors):
            self.upsample_layers.append(
                upsampler(
                    i=i,
                    in_dim=dim,
                    out_dim=dim,
                    attn_dim=hidden_dim,
                    up_factor=up_factor,
                    k=k,
                    radius=radius,
                    interpolation_params=interpolation_params,
                )
            )

        self.predict_normals = predict_normals
        if self.predict_normals:
            self.normal_layer = NormalEstimator(
                in_dim=dim,
                out_dim=dim,
                attn_dim=hidden_dim,
                k=k,
                interpolation_params=interpolation_params,
            )

    def forward(self, partial_xyz, seed_xyz, seed_features):
        """
        Args:
            partial_xyz: partial input coordinates [B, N, 3]
            seed_xyz: Patch Seeds coordinates [B, N1, 3]
            seed_features: Patch Seeds features [B, N1, C]
        Returns:
            completions: list of coarse-to-fine completions of partial shape
                [[B, N1, 3], ..., [B, N4, 3]]
        """
        _, N_seed, _ = seed_xyz.shape

        completions = [seed_xyz]

        # double base number of points by sampling some of the partial input
        xyz = torch.cat([seed_xyz, partial_xyz], dim=1).contiguous()
        idx = fps(xyz, N_seed * 2)
        xyz = index_points(xyz, idx)

        # coarse-to-fine prediction
        features = None
        for i, upsample_layer in enumerate(self.upsample_layers):
            xyz, features = upsample_layer(
                xyz,
                features,
                seed_xyz,
                seed_features,
            )
            completions.append(xyz)

        if self.predict_normals:
            completion_normals = self.normal_layer(
                xyz,
                features,
                seed_xyz,
                seed_features,
            )
        else:
            completion_normals = None

        return completions, completion_normals


class PatchSeedGenerator(nn.Module):
    """
    Patch Seed Generator from SeedFormer for producing 
    Patch Seeds from a set of partial point cloud features.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, global_dim,
                 k=20, up_factor=2, use_tanh=True):
        super().__init__()

        self.up_factor = up_factor
        self.use_tanh = use_tanh

        # feature upsampling/splitting
        self.upsample_transformer = UpsampleTransformer(
            in_dim=in_dim,
            out_dim=out_dim,
            dim=hidden_dim,
            n_knn=k,
            up_factor=up_factor,
            use_upfeat=False,
            scale_layer=None,
            attn_channel=True,
        )

        # seed coordinate regressor
        self.mlp1 = MLP_Res(
            in_dim=out_dim + global_dim,
            hidden_dim=out_dim,
            out_dim=out_dim,
            final_act=False,
        )
        self.mlp2 = MLP_Res(
            in_dim=out_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            final_act=False,
        )
        self.mlp3 = MLP_Res(
            in_dim=out_dim + global_dim,
            hidden_dim=out_dim,
            out_dim=out_dim,
            final_act=False,
        )
        self.mlp4 = MLP(
            in_dim=out_dim,
            out_dim=3,
            hidden_dims=[hidden_dim],
            final_norm=False,
            final_act=False,
        )

    def forward(self, partial_xyz, partial_features, global_feature):
        """
        Args:
            partial_xyz: downsampled sets of xyz coordinates of partial point
                cloud [B, N, 3]
            partial_features: local features corresponding to points
                partial_xyz [B, N, in_dim]
            global_feature: global descriptor of partial shape [B, 1, global_dim]
        Returns:
            seed_xyz: Patch Seeds coordinates [B, N*up_factor, 3]
            seed_features: Patch Seeds features [B, N*up_factor, out_dim]
        """
        _, N, _ = partial_xyz.shape

        # feature upsampling/splitting
        features = self.upsample_transformer(
            pos=partial_xyz,
            key=partial_features,
            query=partial_features,
            upfeat=None,
        )

        # regress seed coordinates
        features = self.mlp1(
            torch.cat(
                [features, global_feature.expand(-1, N*self.up_factor, -1)],
                dim=-1,
            )
        )
        features = self.mlp2(features)
        seed_features = self.mlp3(
            torch.cat(
                [features, global_feature.expand(-1, N*self.up_factor, -1)],
                dim=-1,
            )
        )
        seed_xyz = self.mlp4(seed_features)

        if self.use_tanh:
            seed_xyz = F.tanh(seed_xyz)

        seed_xyz = seed_xyz.contiguous()

        return seed_xyz, seed_features, None


class RobustSeedGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, global_dim, embed_dim, num_blocks=4, 
                 num_freqs=64, num_heads=8, attn_drop=0.0, res_drop=0.0, 
                 up_factor=2):
        super().__init__()

        self.up_factor = up_factor

        # learnable center seed token
        self.center_seed_token = nn.Parameter(torch.zeros(embed_dim))
        self.center_seed_token.data.normal_(mean=0.0, std=0.02)

        # positional encoding
        self.pos_embed= MLP(
            in_dim=3,
            out_dim=embed_dim,
            hidden_dims=[in_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # project feature dimension to transformer token size
        self.input_proj = MLP(
            in_dim=in_dim,
            out_dim=embed_dim,
            hidden_dims=[in_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # transformer blocks with global self attention
        self.transformer_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.transformer_blocks.append(
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    attn_drop,
                    res_drop,
                )
            )

        # upsample transformer block
        self.upsample_block = UpsampleTransformerBlock(
            embed_dim,
            num_heads,
            attn_drop,
            res_drop,
            up_factor,
        )

        # project transformer output tokens to seed feature size
        self.output_proj = MLP(
            in_dim=embed_dim,
            out_dim=out_dim,
            hidden_dims=[embed_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # object center prediction
        self.center_mlp = MLP(
            in_dim=embed_dim + global_dim,
            out_dim=3,
            hidden_dims=[embed_dim, embed_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # predict seed positions relative to object center
        self.offset_mlp = MLP(
            in_dim=out_dim + embed_dim + global_dim,
            out_dim=3,
            hidden_dims=[out_dim, out_dim // 2],
            final_norm=False,
            final_act=False,
        )

    def forward(self, partial_xyz, partial_features, global_feature):
        """
        Args:
            partial_xyz: downsampled sets of xyz coordinates of partial point
                cloud [B, N, 3]
            partial_features: local features corresponding to points
                partial_xyz [B, N, in_dim]
            global_feature: global descriptor of partial shape [B, 1, global_dim]
        Returns:
            seed_xyz: Patch Seeds coordinates [B, N*up_factor, 3]
            seed_features: Patch Seeds features [B, N*up_factor, out_dim]
            object_center: Predicted object center of complete shape [B, 1, 3]
        """
        B, N, _ = partial_features.shape

        # positional encoding [B, N, embed_dim]
        pos_embed = self.pos_embed(partial_xyz)

        # project input features to transformer input tokens [B, N, embed_dim]
        feat_embed = self.input_proj(partial_features)

        center_embed = self.center_seed_token.view(1, 1, -1).repeat(B, 1, 1)
        center_pos_embed = center_embed.new_zeros([B, 1, center_embed.shape[2]])

        feat_embed = torch.cat([feat_embed, center_embed], dim=1)
        pos_embed = torch.cat([pos_embed, center_pos_embed], dim=1)
        embed = pos_embed + feat_embed

        # apply transformer blocks
        for blk in self.transformer_blocks:
            embed = blk(embed)

        center_seed_embed = embed[:, -1:, :]  # [B, 1, embed_dim]
        embed = embed[:, :-1, :]  # [B, N, embed_dim]

        # upsample embedding tokens
        upsampled_embed = self.upsample_block(embed)  # [B, up_factor*N, embed_dim]

        # project upsampled tokens to seed features
        seed_features = self.output_proj(upsampled_embed)  # [B, up_factor*N, out_dim]

        # object center prediction
        center_feat = torch.cat([center_seed_embed, global_feature], dim=-1)
        object_center = self.center_mlp(center_feat)  # [B, 1, 3]

        # predict seed positions relative to object center
        seed_feats = torch.cat([seed_features, center_feat.expand(-1, self.up_factor * N, -1)], dim=-1)
        seed_offsets = self.offset_mlp(seed_feats)  # [B, upfactor*N, 3]
        seed_xyz = object_center + seed_offsets

        seed_xyz = seed_xyz.contiguous()

        return seed_xyz, seed_features, object_center


class ConstrainedSeedGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, global_dim, embed_dim, num_blocks=4, 
                 num_freqs=64, num_heads=8, attn_drop=0.0, res_drop=0.0, 
                 up_factor=2, cross_num_blocks=2, cross_num_heads=8,
                 cross_attn_drop=0.0, cross_res_drop=0.0):
        super().__init__()

        self.up_factor = up_factor

        # learnable center seed token
        self.center_seed_token = nn.Parameter(torch.zeros(embed_dim))
        self.center_seed_token.data.normal_(mean=0.0, std=0.02)

        # free/occupied space tokens
        self.free_space_token = nn.Parameter(torch.zeros(embed_dim)) 
        self.free_space_token.data.normal_(mean=0.0, std=0.02)
        self.occupied_space_token = nn.Parameter(torch.zeros(embed_dim))
        self.occupied_space_token.data.normal_(mean=0.0, std=0.02)

        # positional encoding
        self.pos_embed= MLP(
            in_dim=3,
            out_dim=embed_dim,
            hidden_dims=[in_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # project feature dimension to transformer token size
        self.input_proj = MLP(
            in_dim=in_dim,
            out_dim=embed_dim,
            hidden_dims=[in_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # transformer blocks with global self attention
        self.transformer_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.transformer_blocks.append(
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    attn_drop,
                    res_drop,
                )
            )

        # transformer blocks with cross attention for considering constraints
        self.cross_transformer_blocks = nn.ModuleList()
        for i in range(cross_num_blocks):
            self.cross_transformer_blocks.append(
                CrossTransformerBlock(
                    embed_dim,
                    cross_num_heads,
                    cross_attn_drop,
                    cross_res_drop,
                )
            )

        # upsample transformer block
        self.upsample_block = UpsampleTransformerBlock(
            embed_dim,
            num_heads,
            attn_drop,
            res_drop,
            up_factor,
        )

        # project transformer output tokens to seed feature size
        self.output_proj = MLP(
            in_dim=embed_dim,
            out_dim=out_dim,
            hidden_dims=[embed_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # object center prediction
        self.center_mlp = MLP(
            in_dim=embed_dim + global_dim,
            out_dim=3,
            hidden_dims=[embed_dim, embed_dim // 2],
            final_norm=False,
            final_act=False,
        )

        # predict seed positions relative to object center
        self.offset_mlp = MLP(
            in_dim=out_dim + embed_dim + global_dim,
            out_dim=3,
            hidden_dims=[out_dim, out_dim // 2],
            final_norm=False,
            final_act=False,
        )

    def forward(self, partial_xyz, partial_features, global_feature,
                free_space_points, occupied_space_points, object_transforms):
        """
        Args:
            partial_xyz: downsampled sets of xyz coordinates of partial point
                cloud [B, N, 3]
            partial_features: local features corresponding to points
                partial_xyz [B, N, in_dim]
            global_feature: global descriptor of partial shape [B, 1, global_dim]
            free_space_points: point samples of free space in scene [1, N_free, 3]
            occupied_space_points: point samples of occupied space in scene [1, N_occ, 3]
            object_transforms: per object normalization (scale + translation) [B, 1, 4]
        Returns:
            seed_xyz: Patch Seeds coordinates [B, N*up_factor, 3]
            seed_features: Patch Seeds features [B, N*up_factor, out_dim]
            object_center: Predicted object center of complete shape [B, 1, 3]
        """
        B, N, _ = partial_features.shape
        _, N_free, _ = free_space_points.shape
        _, N_occ, _ = occupied_space_points.shape

        # transform scene constraints into coordinate system of object
        free_space_points = (free_space_points - object_transforms[..., :3]) / object_transforms[..., 3:]
        occupied_space_points = (occupied_space_points - object_transforms[..., :3]) / object_transforms[..., 3:]

        # scene constraint learnable tokens
        free_space_embed = self.free_space_token.view(1, 1, -1).repeat(B, N_free, 1)
        occupied_space_embed = self.occupied_space_token.view(1, 1, -1).repeat(B, N_occ, 1)
        constraint_space_embed = torch.cat([free_space_embed, occupied_space_embed], dim=1)

        # scene constraint positional encoding [B, N_free+N_occ, embed_dim]
        constraint_points = torch.cat([free_space_points, occupied_space_points], dim=1)
        constraint_pos_embed = self.pos_embed(constraint_points)

        # construct scene constraint embedding
        constraint_embed = constraint_pos_embed + constraint_space_embed 

        # object positional encoding [B, N, embed_dim]
        pos_embed = self.pos_embed(partial_xyz)

        # project object input features to transformer input tokens [B, N, embed_dim]
        feat_embed = self.input_proj(partial_features)

        # object center learnable token
        center_embed = self.center_seed_token.view(1, 1, -1).repeat(B, 1, 1)
        center_pos_embed = center_embed.new_zeros([B, 1, center_embed.shape[2]])

        # construct object embedding
        feat_embed = torch.cat([feat_embed, center_embed], dim=1)
        pos_embed = torch.cat([pos_embed, center_pos_embed], dim=1)
        obj_embed = pos_embed + feat_embed

        # apply transformer blocks with self attention (partial object only)
        embed1 = obj_embed
        for blk in self.transformer_blocks:
            embed1 = blk(embed1)

        # apply transformer blocks with cross attention (consider known scene constraints)
        embed2 = obj_embed
        for blk in self.cross_transformer_blocks:
            embed2 = blk(embed2, constraint_embed)

        embed = embed1 + embed2
        center_seed_embed = embed[:, -1:, :]  # [B, 1, embed_dim]
        embed = embed[:, :-1, :]  # [B, N, embed_dim]

        # upsample embedding tokens
        upsampled_embed = self.upsample_block(embed)  # [B, up_factor*N, embed_dim]

        # project upsampled tokens to seed features
        seed_features = self.output_proj(upsampled_embed)  # [B, up_factor*N, out_dim]

        # object center prediction
        center_feat = torch.cat([center_seed_embed, global_feature], dim=-1)
        object_center = self.center_mlp(center_feat)  # [B, 1, 3]

        # predict seed positions relative to object center
        seed_feats = torch.cat([seed_features, center_feat.expand(-1, self.up_factor * N, -1)], dim=-1)
        seed_offsets = self.offset_mlp(seed_feats)  # [B, upfactor*N, 3]
        seed_xyz = object_center + seed_offsets

        seed_xyz = seed_xyz.contiguous()

        return seed_xyz, seed_features, object_center

