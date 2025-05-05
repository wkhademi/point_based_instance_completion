import torch
import torch.nn.functional as F

from libs import chamfer_distance
from pointnet2_ops.pointnet2_utils import gather


def completion_loss(pred, gt, pred_normals=None, gt_normals=None):
    dist1, dist2, idx1, idx2 = chamfer_distance(pred, gt)

    # chamfer distance
    d1 = torch.sqrt(torch.clamp(dist1, min=1e-9))
    d2 = torch.sqrt(torch.clamp(dist2, min=1e-9))
    loss = 0.5 * (torch.mean(d1) + torch.mean(d2))

    # compute normal loss if normals were predicted
    if pred_normals is not None:
        gt_normals = gt_normals.permute(0, 2, 1).contiguous()
        assigned_normals = gather(gt_normals, idx1).permute(0, 2, 1)
        normal_loss = torch.mean(1. - F.cosine_similarity(pred_normals, assigned_normals, dim=-1))

        return loss, normal_loss

    return loss, torch.tensor(0., device=pred.device)


def partial_loss(pred, gt):
    dist1, _, _, _ = chamfer_distance(pred, gt)

    d1 = torch.sqrt(torch.clamp(dist1, min=1e-9))
    loss = torch.mean(d1)

    return loss