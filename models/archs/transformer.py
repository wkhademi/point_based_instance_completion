import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum

from utils.pc_utils import knn, index_points
from models.archs.mlp import MLP_Res
from models.archs.layer_utils import (
    BatchNorm2d, ConvTranspose1d, ConvTranspose2d, Upsample
)


class UpsampleTransformer(nn.Module):
    """
    Upsample Transformer module proposed in SeedFormer: Patch Seeds based
    Point Cloud Completion with Upsample Transformer [Zhou et al., ECCV 2022].
    """
    def __init__(self, in_dim, out_dim, dim, n_knn=20, up_factor=2,
                 use_upfeat=True, pos_hidden_dim=64, attn_hidden_multiplier=4,
                 scale_layer=nn.Softmax, attn_channel=True):
        super().__init__()

        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_dim = dim if attn_channel else 1

        # Q, K, V linear projection
        self.mlp_v = MLP_Res(
            in_dim=in_dim * 2,
            hidden_dim=in_dim,
            out_dim=in_dim,
            final_act=False,
        )
        self.q_proj = nn.Linear(in_dim, dim)
        self.k_proj = nn.Linear(in_dim, dim)
        self.v_proj = nn.Linear(in_dim, dim)

        if use_upfeat:
            self.upfeat = nn.Linear(in_dim, dim)

        # relative positional encoder
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_hidden_dim),
            BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_hidden_dim, dim),
        )

        # attention layers
        self.attn_mlp = [
            nn.Linear(dim, dim * attn_hidden_multiplier),
            BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU()
        ]

        if up_factor:
            self.attn_mlp.append(
                ConvTranspose2d(
                    dim * attn_hidden_multiplier,
                    attn_out_dim,
                    (up_factor,1),
                    (up_factor,1),
                )
            )
        else:
            self.attn_mlp.append(
                nn.Linear(
                    dim * attn_hidden_multiplier,
                    attn_out_dim,
                )
            )

        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # attention weight scaling
        self.scale = scale_layer(
            dim=-2
        ) if scale_layer is not None else nn.Identity()

        # upsample values and identity features
        self.upsample_value = Upsample(
            scale_factor=(up_factor, 1)
        ) if up_factor else nn.Identity()
        self.upsample_identity = Upsample(
            scale_factor=up_factor
        ) if up_factor else nn.Identity()

        # residual connection
        self.linear_end = nn.Linear(dim, out_dim)
        if in_dim != out_dim:
            self.identity_layer = nn.Linear(in_dim, out_dim)
        else:
            self.identity_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat=None):
        """
        Args:
            pos: xyz coordinates of a set of points [B, N, 3]
            key: a set of key features associated with pos [B, N, in_dim]
            query: a set of query features associated with pos [B, N, in_dim]
            upfeat: interpolated features from Patch Seeds [B, N, in_dim]
        Returns:
            upsampled_features: a set of upsampled features upsampled by a
                factor of up_factor [B, N*up_factor, out_dim]
        """
        # Q, K, V linear projection
        value = self.mlp_v(torch.cat([key, query], -1))  # (B, N, dim)
        identity = value
        key = self.k_proj(key)  # (B, N, dim)
        query = self.q_proj(query)
        value = self.v_proj(value)

        B, N, _ = value.shape

        # compute neighborhoods to perform attention over
        idx = knn(pos, pos, k=self.n_knn)

        # relative positional embedding (B, N, k, dim)
        pos_rel = pos.view(B, N, 1, -1) - index_points(pos, idx)
        pos_embedding = self.pos_mlp(pos_rel)

        # relative difference between query and key features (B, N, k, dim)
        qk_rel = query.view(B, N, 1, -1) - index_points(key, idx)

        # upfeat embedding
        if self.use_upfeat:
            upfeat = self.upfeat(upfeat)  # (B, N, dim)
            upfeat_rel = (
                upfeat.reshape((B, N, 1, -1)) - index_points(upfeat, idx)
            )  # (B, N, k, dim)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        # compute attention (B, N*up_factor, k, dim)
        attention = self.scale(
            self.attn_mlp(pos_embedding + qk_rel + upfeat_rel)
        )

        # upsample values
        value = (
            index_points(value, idx) + pos_embedding + upfeat_rel
        ) # (B, N, k, dim)
        value = self.upsample_value(value)  # (B, N*up_factor, k, dim)

        # compute result of local attention
        agg = einsum('b i j c, b i j c -> b i c', attention, value)
        residual = self.linear_end(agg)  # (B, N*up_factor, out_dim)

        # identity shortcut (+ upsample)
        identity = self.identity_layer(identity)  # (B, N, out_dim)
        identity = self.upsample_identity(identity)  # (B, N*up_factor, out_dim)

        upsampled_features = identity + residual

        return upsampled_features


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop, res_drop):
        super().__init__()

        self.num_heads = num_heads
        self.attn_drop = attn_drop

        # pre-attention layer norm
        self.pre_norm = nn.LayerNorm(embed_dim)

        # projection layers for key, query, value, across all attention heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # attention output layer
        self.res_proj = nn.Linear(embed_dim, embed_dim)

        # dropout layers
        self.res_dropout = nn.Dropout(res_drop)

        # post-attention layer norm
        self.post_norm = nn.LayerNorm(embed_dim)

        # post-attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(res_drop),
        )

    def forward(self, x):
        B, N, embed_dim = x.shape

        # linear projection of queries/keys/values
        qkv = self.qkv_proj(self.pre_norm(x))
        query, key, value = qkv.chunk(3, -1)
        query = query.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        key = key.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        value = value.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # dot product attention
        y = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.attn_drop
        )
        y = y.permute(0, 2, 1, 3).view(B, N, -1)

        # post attention residuals
        x = x + self.res_dropout(self.res_proj(y))
        x = x + self.mlp(self.post_norm(x))

        return x


class CrossTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop, res_drop):
        super().__init__()

        self.num_heads = num_heads
        self.attn_drop = attn_drop

        # pre-attention layer norm
        self.pre_norm = nn.LayerNorm(embed_dim)

        # projection layers for key, query, value, across all attention heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # attention output layer
        self.res_proj = nn.Linear(embed_dim, embed_dim)

        # dropout layers
        self.res_dropout = nn.Dropout(res_drop)

        # post-attention layer norm
        self.post_norm = nn.LayerNorm(embed_dim)

        # post-attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(res_drop),
        )

    def forward(self, x, y):
        B, N, embed_dim = x.shape
        _, M, _ = y.shape

        # linear projection of queries/keys/values
        query = self.q_proj(self.pre_norm(x))
        key = self.k_proj(self.pre_norm(y))
        value = self.v_proj(self.pre_norm(y))
        query = query.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        key = key.view(B, M, self.num_heads, -1).permute(0, 2, 1, 3)
        value = value.view(B, M, self.num_heads, -1).permute(0, 2, 1, 3)

        # dot product attention
        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.attn_drop
        )
        out = out.permute(0, 2, 1, 3).view(B, N, -1)

        # post attention residuals
        x = x + self.res_dropout(self.res_proj(out))
        x = x + self.mlp(self.post_norm(x))

        return x


class UpsampleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop, res_drop, up_factor, softmax=False):
        super().__init__()

        self.num_heads = num_heads
        self.up_factor = up_factor
        self.softmax = softmax
        self.attn_drop = attn_drop

        # pre-attention layer norm
        self.pre_norm = nn.LayerNorm(embed_dim)

        # projection layers for key, query, value, across all attention heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # upsample layer
        self.upsample_queries = ConvTranspose1d(
            embed_dim, embed_dim, up_factor, up_factor
        )
        self.upsample_identity = Upsample(scale_factor=up_factor)

        # attention output layer
        self.res_proj = nn.Linear(embed_dim, embed_dim)

        # dropout layers
        self.attn_dropout = nn.Dropout(attn_drop)
        self.res_dropout = nn.Dropout(res_drop)

        # post-attention layer norm
        self.post_norm = nn.LayerNorm(embed_dim)

        # post-attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(res_drop),
        )

    def forward(self, x):
        B, N, embed_dim = x.shape

        # linear projection of queries/keys/values
        qkv = self.qkv_proj(self.pre_norm(x))
        query, key, value = qkv.chunk(3, -1)
        key = key.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        value = value.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # upsample query tokens
        query = self.upsample_queries(query)  # [B, up_factor*N, embed_dim]
        query = query.view(B, self.up_factor * N, self.num_heads, -1).permute(0, 2, 1, 3)

        # dot product attention
        if self.softmax:
            y = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.attn_drop
            )
        else:
            scale_factor = 1. / math.sqrt(query.size(-1))
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight = self.attn_dropout(attn_weight)
            y = attn_weight @ value  # [B, num_heads, up_factor*N, head_dim]
        y = y.permute(0, 2, 1, 3).reshape(B, self.up_factor * N, embed_dim)

        # post attention residuals
        x = self.upsample_identity(x)
        x = x + self.res_dropout(self.res_proj(y))
        x = x + self.mlp(self.post_norm(x))

        return x