import os
import nksr
import torch
import trimesh
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from utils.pc_utils import batch_downsample
from utils.log_utils import log_scene_visualizations
from utils.loss_utils import completion_loss, partial_loss


SHAPENETCLASSES = ['void',
                   'table', 'jar', 'skateboard', 'car', 'bottle',
                   'tower', 'chair', 'bookshelf', 'camera', 'airplane',
                   'laptop', 'basket', 'sofa', 'knife', 'can',
                   'rifle', 'train', 'pillow', 'lamp', 'trash_bin',
                   'mailbox', 'watercraft', 'motorbike', 'dishwasher', 'bench',
                   'pistol', 'rocket', 'loudspeaker', 'file cabinet', 'bag',
                   'cabinet', 'bed', 'birdhouse', 'display', 'piano',
                   'earphone', 'telephone', 'stove', 'microphone', 'bus',
                   'mug', 'remote', 'bathtub', 'bowl', 'keyboard',
                   'guitar', 'washer', 'bicycle', 'faucet', 'printer',
                   'cap', 'clock', 'helmet', 'flowerpot', 'microwaves']


def save_point_cloud(
    pc, 
    normals, 
    object_transforms, 
    save_path, 
    classes=None, 
    scores=None,
):
    pc = ((pc * object_transforms[..., 3:]) + object_transforms[..., :3]).reshape(-1, 3)
    normals = normals.reshape(-1, 3)

    if classes is None and scores is None:
        np.savez(save_path, xyz=pc, normal=normals)
    else:
        np.savez(save_path, xyz=pc, normal=normals, classes=classes, scores=scores)


def save_meshes(
    object_meshes,
    save_dir,
    scene_name,
    classes=None,
    scores=None,
):
    for idx, object_mesh in enumerate(object_meshes):
        if classes is None and scores is None:
            file_name = f"{scene_name}_{idx}.ply"
        else:
            file_name = f"{scene_name}_{idx}_{SHAPENETCLASSES[classes[idx]]}_{scores[idx]}.ply"

        save_path = os.path.join(save_dir, "reconstructions", file_name)
        object_mesh.export(save_path)


def save(
    partial_points,
    partial_normals,
    pred_points,
    pred_normals,
    object_transforms,
    pred_meshes,
    save_dir,
    scene_name,
    classes,
    scores,
):
    os.makedirs(os.path.join(save_dir, "partials"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "completions"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "reconstructions"), exist_ok=True)

    # save partial scan
    save_point_cloud(
        partial_points,
        partial_normals,
        object_transforms,
        os.path.join(save_dir, f"partials/{scene_name}.npz"),
    )

    # save predicted point clouds
    save_point_cloud(
        pred_points,
        pred_normals,
        object_transforms,
        os.path.join(save_dir, f"completions/{scene_name}.npz"),
        classes,
        scores,
    )

    # save predicted meshes
    save_meshes(
        pred_meshes,
        save_dir,
        scene_name,
        classes,
        scores,
    )


def reconstruct_mesh(reconstructor, points, normals, num_points_per_obj):
    num_objs = int(points.shape[0] / num_points_per_obj)

    meshes = []
    for i in range(num_objs):
        input_xyz = points[num_points_per_obj * i : num_points_per_obj * (i + 1)]
        input_normal = normals[num_points_per_obj * i : num_points_per_obj * (i + 1)]
        
        # reconstruct object mesh from points and surface normals
        field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=1.0)
        mesh = field.extract_dual_mesh(mise_iter=1)
        verts = mesh.v.clone().detach().cpu().numpy()
        faces = mesh.f.clone().detach().cpu().numpy()
        reconstructed_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        meshes.append(reconstructed_mesh)

    return meshes


def test(
    test_dataloader, 
    model,  
    logger, 
    tensorboard_logger,
    experiment_dir,
    cfg,
):
    logger.info("Testing scene-level completion model...")

    # create NKSR mesh reconstructor
    reconstructor = nksr.Reconstructor(device="cuda", config="snet")

    model.eval()
    with torch.no_grad():
        avg_L_cd1 = []
        avg_L_cd2 = []
        avg_L_cd3 = []
        avg_L_cd4 = []
        avg_L_part = []
        avg_L_center = []
        avg_L_normal = []

        for idx, data in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
            scene_id = data["scene_id"][0]
            partial = data["partial"][0].cuda()
            partial_normals = data["partial_normals"][0].cuda()
            free_space_points = data["free_space"].cuda()
            occupied_space_points = data["occupied_space"].cuda()
            object_transforms = data["transforms"][0].cuda()
            
            if cfg.test_dataloader.dataset.dataset_name == "ScanWCF":
                gt_completions = data["complete"][0].cuda()
                gt_normals = data["complete_normals"][0].cuda()
                gt_center = data["object_center"][0].cuda()
                partial_id = data["partial_id"][0].item()
                classes = None
                scores = None
            elif cfg.test_dataloader.dataset.dataset_name == "Mask3D":
                partial_id = data["partial_id"][0].item()
                classes = data["classes"][0].detach().cpu().numpy()
                scores = data["scores"][0].detach().cpu().numpy()

            # subsample partial point cloud
            downsampled_partials, downsampled_partial_normals = batch_downsample(
                partial, 
                n_samples=cfg.test_dataloader.input_downsample_points,
                normals=partial_normals,
            )
            partials = [partial] + downsampled_partials
            partial_normals = [partial_normals] + downsampled_partial_normals

            # produce completions
            pred_completions, pred_normals, pred_center = model(
                partials, 
                partial_normals, 
                free_space_points, 
                occupied_space_points,
                object_transforms,
            )
            pred_normals = F.normalize(pred_normals, dim=-1)

            # reconstruct meshes
            num_points_per_obj = cfg.test_dataloader.num_gt_points
            pred_points = (
                (pred_completions[3] * object_transforms[..., 3:]) + object_transforms[..., :3]
            )
            pred_meshes = reconstruct_mesh(
                reconstructor, 
                pred_points.reshape(-1, 3), 
                pred_normals.reshape(-1, 3), 
                num_points_per_obj,
            )

            # compute metrics when using ground truth instance segmentations
            if cfg.test_dataloader.dataset.dataset_name == "ScanWCF":
                # subsample ground truth point cloud
                downsampled_gt_completions = batch_downsample(
                    gt_completions, 
                    n_samples=cfg.test_dataloader.gt_downsample_points,
                )

                # compute completion losses
                L_cd1, _ = completion_loss(pred_completions[0], downsampled_gt_completions[2]) 
                L_cd2, _ = completion_loss(pred_completions[1], downsampled_gt_completions[1])
                L_cd3, _ = completion_loss(pred_completions[2], downsampled_gt_completions[0])
                L_cd4, L_normal = completion_loss(pred_completions[3], gt_completions, pred_normals, gt_normals)

                # compute partial reconstruction loss
                L_part = partial_loss(partials[0], pred_completions[3])

                # compute object center loss
                L_center = F.mse_loss(pred_center, gt_center)

                avg_L_cd1.append(L_cd1.item())
                avg_L_cd2.append(L_cd2.item())
                avg_L_cd3.append(L_cd3.item())
                avg_L_cd4.append(L_cd4.item())
                avg_L_part.append(L_part.item())
                avg_L_center.append(L_center.item())
                avg_L_normal.append(L_normal.item())

                # visualize point clouds
                log_scene_visualizations(
                    partial.detach().cpu(), 
                    pred_completions[3].clone().detach().cpu(), 
                    gt_completions.detach().cpu(), 
                    object_transforms.detach().cpu(),
                    test_dataloader.dataset.instance_colors,
                    tensorboard_logger, 
                    idx, 
                    "Test",
                )

            # save completions
            save(
                partial.detach().cpu().numpy(), 
                partial_normals[0].detach().cpu().numpy(), 
                pred_completions[3].detach().cpu().numpy(), 
                pred_normals.detach().cpu().numpy(), 
                object_transforms.detach().cpu().numpy(),
                pred_meshes,
                os.path.abspath(os.path.join(experiment_dir, f"results_{cfg.test_dataloader.dataset.dataset_name}")),
                scene_id + "_" + str(partial_id),
                classes,
                scores,
            )

        # log validation losses and visualization
        losses = {
            "Test L_cd1": np.mean(avg_L_cd1),
            "Test L_cd2": np.mean(avg_L_cd2),
            "Test L_cd3": np.mean(avg_L_cd3),
            "Test L_cd4": np.mean(avg_L_cd4),
            "Test L_part": np.mean(avg_L_part),
            "Test L_center": np.mean(avg_L_center),
            "Test L_normal": np.mean(avg_L_normal),
        }
        for name, loss in losses.items():
            logger.info(f"{name}: {str(loss)}")