import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pc_utils import knn, index_points
from models.archs.mlp import MLP


def viewpoint_invariant_transform(grouped_xyz_rel, normals, new_normals, nn_idx):
    """
    Compute the viewpoint-invariant relative positional encoding from 
    Improving the Robustness of Point Convolution on k-Nearest Neighbor 
    Neighborhoods with a Viewpoint-Invariant Coordinate Transform 
    [Li et al., WACV 2023].

    Args:
        grouped_xyz_rel: [B, S, k, 3]
        normals: surface normals associated with xyz coordinates [B, N, 3]
        new_normals: surface normals associated with new_xyz coordinates [B, S, 3]
        nn_idx: indices describing neighborhoods of grouped_xyz_rel [B, S, k]
        k: number of nearest neighbors
    Returns:
        grouped_vi_rel: concatentation of rotation+scale invariant descriptors,
                        rotation invariant descriptors, and non-invariant 
                        descriptors [B, S, k, 11]
    """
    B, S, _ = new_normals.shape

    # construct orthonormal basis
    r_hat = F.normalize(grouped_xyz_rel, dim=-1)  # [B, S, k, 3]
    v_mu = (
        new_normals.view(B, S, 1, 3) - 
        torch.sum(
            new_normals.view(B, S, 1, 3) * r_hat, dim=-1, keepdim=True
        ) * r_hat
    )  # [B, S, k, 3]
    v_mu = F.normalize(v_mu, dim=-1)  
    w_mu = torch.cross(r_hat, v_mu, dim=-1)  # [B, S, k, 3]
    w_mu = F.normalize(w_mu, dim=-1)

    # group the normals of neighborhood
    grouped_normals = index_points(normals, nn_idx)  # [B, S, k, 3]

    # construct rotation+scale invariant descriptors
    theta1 = torch.sum(
        grouped_normals * new_normals.view(B, S, 1, 3), dim=-1, keepdim=True
    )
    theta2 = torch.sum(
        r_hat * new_normals.view(B, S, 1, 3), dim=-1, keepdim=True
    )
    theta3 = torch.sum(r_hat * grouped_normals, dim=-1, keepdim=True)
    theta4 = torch.sum(v_mu * grouped_normals, dim=-1, keepdim=True)
    theta5 = torch.sum(w_mu * grouped_normals, dim=-1, keepdim=True)

    # construct rotation invariant only descriptors
    theta6 = torch.sum(
        grouped_xyz_rel * new_normals.view(B, S, 1, 3), dim=-1, keepdim=True
    )
    theta7 = torch.sum(
        grouped_xyz_rel * 
        torch.cross(
            grouped_normals,
            new_normals.view(B, S, 1, 3).expand_as(grouped_normals),
            dim=-1,
        ),
        dim=-1,
        keepdim=True,
    )
    theta8 = grouped_xyz_rel.norm(dim=-1, keepdim=True)

    grouped_vi_rel = torch.cat(
        [
            theta1,
            theta2,
            theta3,
            theta4,
            theta5,
            theta6,
            theta7,
            theta8,
            grouped_xyz_rel,
        ], dim=-1
    ).contiguous()  # [B, S, k, 11]

    return grouped_vi_rel


def group_neighborhood(xyz, features, new_xyz, normals=None, 
                       new_normals=None, nn_idx=None, k=16, use_vi=False):
    """
    Args:
        xyz: xyz coordinates of a set of points [B, N, 3]
        features: per point feature corresponding to points xyz [B, N, C]
        new_xyz: xyz coordinates of a set of points [B, S, 3]
        normals: surface normals associated with xyz coordinates [B, N, 3]
        new_normals: surface normals associated with new_xyz coordinates [B, S, 3]
        nn_idx (optional): indices of points in xyz that are the nearest
            neighbors of points in new_xyz [B, S, k]
        k (optional, default=16): number of nearest neighbors to sample
            per point
        use_vi: construct viewpoint invariant relative positional encoding
    Returns:
        grouped_features: features of grouped nearest neighbors [B, S, k, C]
        weightnet_input: grouped neighborhood for input to WeightNet [B, S, k, D]
    """
    B, S, _ = new_xyz.shape

    # find nearest neighbors if not provided
    if nn_idx is None:
        idx = knn(new_xyz, xyz, k=k)
    else:
        idx = nn_idx

    # group coordinates of nearest neighbors
    grouped_xyz = index_points(xyz, idx)  # [B, S, k, 3]

    # compute the relative position of each grouped coordinate
    grouped_xyz_rel = grouped_xyz - new_xyz.view(B, S, 1, 3)

    # group features as well if provided [B, S, k, C]
    if features is not None:
        grouped_features = index_points(features, idx)
    else:
        grouped_features = grouped_xyz_rel

    # construct viewpoint invariant descriptor [B, S, k, D]
    if use_vi:
        weightnet_input = viewpoint_invariant_transform(
            grouped_xyz_rel,
            normals,
            new_normals,
            idx,
        )
    else:
        weightnet_input = grouped_xyz_rel

    return grouped_features, weightnet_input


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super().__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Linear(in_channel, out_channel))
            self.mlp_norms.append(nn.LayerNorm(out_channel))
        else:
            self.mlp_convs.append(nn.Linear(in_channel, hidden_unit[0]))
            self.mlp_norms.append(nn.LayerNorm(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Linear(hidden_unit[i-1], hidden_unit[i]))
                self.mlp_norms.append(nn.LayerNorm(hidden_unit[i]))
            self.mlp_convs.append(nn.Linear(hidden_unit[i], out_channel))
            self.mlp_norms.append(nn.LayerNorm(out_channel))

    def forward(self, input):
        weights = input
        for i, conv in enumerate(self.mlp_convs):
            norm = self.mlp_norms[i]
            weights = F.leaky_relu(norm(conv(weights)), 0.2)

        return weights


class PointConv(nn.Module):
    """
    PointConv module from PointConv: Deep Convolutional Networks on 3D Point
    Clouds [Wu et al., CVPR 2019].
    """
    def __init__(self, k, point_dim, in_channel, out_channel, mlp=[], 
                 c_mid=16, norm=True, act=True, use_vi=False):
        super().__init__()

        self.k = k
        self.norm = norm
        self.act = act
        self.use_vi = use_vi

        # viewpoint invariant transform adds 8 extra inputs
        if self.use_vi:
            point_dim += 8

        feat_dim = point_dim

        self.mlp_convs = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        in_dim = point_dim
        for dim in mlp:
            self.mlp_convs.append(nn.Linear(in_dim, dim))
            self.mlp_norms.append(nn.LayerNorm(dim))
            in_dim = dim

        if len(mlp) != 0:
            feat_dim = mlp[-1]

        last_ch = feat_dim + in_channel

        self.weightnet = WeightNet(point_dim, c_mid)
        self.linear = nn.Linear(c_mid * last_ch, out_channel)
        if self.norm:
            self.norm_linear = nn.LayerNorm(out_channel)

    def forward(self, xyz, features, new_xyz, normals=None, 
                new_normals=None, nn_idx=None):
        """
        Args:
            xyz: xyz coordinates of a set of points
            features: per point feature corresponding to points xyz
            new_xyz: xyz coordinates of a set of points without corresponding
                features
            normals: surface normals associated with xyz coordinates
            new_normals: surface normals associated with new_xyz coordinates
            nn_idx (optional): indices of points in xyz that are the nearest
                neighbors of points in new_xyz
        Return:
            new_features: per point features corresponding to points new_xyz
        """
        if self.use_vi:
            assert normals is not None, \
                "must provide normals for VI-PointConv"
            assert new_normals is not None, \
                "must provide new_normals for VI-PointConv"

        B, _, _ = xyz.shape
        _, S, _ = new_xyz.shape

        # group features in neighborhood
        grouped_features, weightnet_input = group_neighborhood(
            xyz, 
            features, 
            new_xyz, 
            normals=normals, 
            new_normals=new_normals, 
            nn_idx=nn_idx, 
            k=self.k,
            use_vi=self.use_vi,
        )

        rel_feats = weightnet_input  # [B, S, k, point_dim]
        for i, conv in enumerate(self.mlp_convs):
            norm = self.mlp_norms[i]
            rel_feats = F.leaky_relu(norm(conv(rel_feats)), 0.2)

        grouped_features = torch.cat([grouped_features, rel_feats], dim=-1)  # [B, S, k, last_ch]

        # convolution on neighborhood
        weights = self.weightnet(weightnet_input)  # [B, S, k, c_mid]
        new_features = (
            grouped_features.permute(0, 1, 3, 2) @ weights
        ).view(B, S, -1)  # [B, S, c_mid * last_ch]
        new_features = self.linear(new_features)  # [B, S, out_channel]

        if self.norm:
            new_features = self.norm_linear(new_features)

        if self.act:
            new_features = F.leaky_relu(new_features, 0.2)

        return new_features


class BottleneckPointConv(nn.Module):
    """
    Bottlenecked version of PointConv
    """
    def __init__(self, k, point_dim, in_channel, out_channel,
                 mlp=[], bottleneck=4, c_mid=16, use_vi=False):
        super().__init__()

        self.reduce = MLP(in_dim=in_channel, out_dim=in_channel//bottleneck)

        self.pointconv = PointConv(
            k=k,
            point_dim=point_dim,
            in_channel=in_channel//bottleneck,
            out_channel=out_channel//bottleneck,
            mlp=mlp,
            c_mid=c_mid,
            use_vi=use_vi,
        )

        self.expand = MLP(
            in_dim=out_channel//bottleneck,
            out_dim=out_channel,
            final_act=False,
        )

        if in_channel != out_channel:
            self.residual_layer = nn.Linear(in_channel, out_channel)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, xyz, features, normals=None, nn_idx=None):
        """
        Args:
            xyz: xyz coordinates of a set of points
            features: per point feature corresponding to points xyz
            normals: surface normals associated with xyz coordinates
            nn_idx (optional): nearest neighbor indices of points in xyz
        Returns:
            new_features: updated per point features corresponding to points xyz
        """
        reduced_features = self.reduce(features)

        new_features = self.pointconv(
            xyz, 
            reduced_features, 
            xyz, 
            normals=normals,
            new_normals=normals,
            nn_idx=nn_idx,
        )

        new_features = self.expand(new_features)

        shortcut = self.residual_layer(features)

        new_features = F.leaky_relu(new_features + shortcut, 0.2)

        return new_features


class BottleneckInterpPointConv(nn.Module):
    """
    Bottlenecked PointConv module meant for interpolating features for a set of
    new xyz coordinates from a set of existing xyz coordinates and their 
    corresponding features.
    """
    def __init__(self, k, point_dim, in_channel, out_channel,
                 mlp=[], bottleneck=4, c_mid=16, use_vi=False):
        super().__init__()

        self.k = k

        self.reduce = MLP(in_dim=in_channel, out_dim=in_channel//bottleneck)

        self.pointconv = PointConv(
            k=k,
            point_dim=point_dim,
            in_channel=in_channel//bottleneck,
            out_channel=out_channel//bottleneck,
            mlp=mlp,
            c_mid=c_mid,
            use_vi=use_vi,
        )

        self.expand = MLP(
            in_dim=out_channel//bottleneck,
            out_dim=out_channel,
            final_act=False,
        )

        if in_channel != out_channel:
            self.residual_layer = nn.Linear(in_channel, out_channel)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, xyz, features, new_xyz, normals=None,
                new_normals=None, nn_idx=None):
        """
        Args:
            xyz: xyz coordinates of a set of points
            features: per point feature corresponding to points xyz
            new_xyz: xyz coordinates of a set of points without corresponding
                features
            normals: surface normals associated with xyz coordinates
            new_normals: surface normals associated with new_xyz coordinates
            nn_idx (optional): indices of points in xyz that are the nearest
                neighbors of points in new_xyz
        Returns:
            new_features: per point interpolated feature corresponding to
                points new_xyz
        """
        # find nearest neighbors of new_xyz in xyz if not provided
        if nn_idx is None:
            nn_idx = knn(new_xyz, xyz, k=self.k)

        reduced_features = self.reduce(features)

        # interpolate features for new_xyz
        new_features = self.pointconv(
            xyz, 
            reduced_features, 
            new_xyz, 
            normals=normals,
            new_normals=new_normals,
            nn_idx=nn_idx,
        )

        new_features = self.expand(new_features)

        grouped_features = index_points(features, nn_idx)
        features = torch.mean(grouped_features, dim=2)
        shortcut = self.residual_layer(features)

        new_features = F.leaky_relu(new_features + shortcut, 0.2)

        return new_features