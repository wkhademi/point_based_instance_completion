import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.io_utils import save_model_ckpt
from utils.pc_utils import batch_downsample
from utils.loss_utils import completion_loss, partial_loss
from utils.log_utils import log_losses, log_scene_visualizations, log_visualizations


def train(
    dataloaders, 
    model, 
    optimizer, 
    scheduler, 
    logger, 
    tensorboard_logger,
    experiment_dir,
    cfg,
    start_epoch=0,
    best_metric_val=1e10,
):
    (train_dataloader, val_dataloader) = dataloaders

    logger.info("Training scene-level completion model...")

    new_best = False
    for epoch in range(start_epoch, cfg.num_epochs):
        avg_L_cd1 = []
        avg_L_cd2 = []
        avg_L_cd3 = []
        avg_L_cd4 = []
        avg_L_part = []
        avg_L_center = []
        avg_L_normal = []

        scheduler.step()
        model.train()

        for data in tqdm(train_dataloader, total=len(train_dataloader)):
            partial = data["partial"].cuda()
            partial_normals = data["partial_normals"].cuda()
            gt_completions = data["complete"].cuda()
            gt_normals = data["complete_normals"].cuda()
            gt_center = data["object_center"].cuda()
            free_space_points = data["free_space"].cuda()
            occupied_space_points = data["occupied_space"].cuda()
            object_transforms = data["transforms"].cuda()

            # subsample partial point cloud
            downsampled_partials, downsampled_partial_normals = batch_downsample(
                partial, 
                n_samples=cfg.train_dataloader.input_downsample_points,
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

            # subsample ground truth point cloud
            downsampled_gt_completions = batch_downsample(
                gt_completions, 
                n_samples=cfg.train_dataloader.gt_downsample_points
            )

            # compute completion and normal losses
            L_cd1, _ = completion_loss(pred_completions[0], downsampled_gt_completions[2])
            L_cd2, _ = completion_loss(pred_completions[1], downsampled_gt_completions[1])
            L_cd3, _ = completion_loss(pred_completions[2], downsampled_gt_completions[0])
            L_cd4, L_normal = completion_loss(pred_completions[3], gt_completions, pred_normals, gt_normals)
            loss = cfg.lam_comp * (L_cd1 + L_cd2 + L_cd3 + L_cd4) + cfg.lam_normal * L_normal

            # compute partial reconstruction loss
            L_part = partial_loss(partials[0], pred_completions[3])
            loss += cfg.lam_part * L_part
                
            # compute object center loss
            L_center = F.mse_loss(pred_center, gt_center)
            loss += cfg.lam_center * L_center

            # update network parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_L_cd1.append(L_cd1.item())
            avg_L_cd2.append(L_cd2.item())
            avg_L_cd3.append(L_cd3.item())
            avg_L_cd4.append(L_cd4.item())
            avg_L_part.append(L_part.item())
            avg_L_center.append(L_center.item())
            avg_L_normal.append(L_normal.item())

        # log average losses and visualizations
        losses = {
            "L_cd1": np.mean(avg_L_cd1),
            "L_cd2": np.mean(avg_L_cd2),
            "L_cd3": np.mean(avg_L_cd3),
            "L_cd4": np.mean(avg_L_cd4),
            "L_part": np.mean(avg_L_part),
            "L_center": np.mean(avg_L_center),
            "L_normal": np.mean(avg_L_normal),
        }
        log_losses(losses, logger, tensorboard_logger, epoch, cfg.num_epochs)
        log_visualizations(
            partials, 
            pred_completions, 
            gt_completions, 
            tensorboard_logger, 
            epoch, 
            "Train",
        )

        # evaluate model on valdiation set
        if epoch == 0 or (epoch + 1) % cfg.validate_freq == 0:
            metric_val = validate(
                val_dataloader, model, logger, tensorboard_logger, cfg, epoch
            )

            if metric_val < best_metric_val:
                best_metric_val = metric_val
                new_best = True
            else:
                new_best = False
        else:
            new_best = False

        # save model weights
        if epoch == 0 or (epoch + 1) % cfg.save_model_freq == 0:
            m = model.module if isinstance(model, nn.DataParallel) else model

            state = {
                "epoch": epoch + 1,
                "best_metric_val": best_metric_val,
                "model_state_dict": m.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            save_model_ckpt(state, experiment_dir, logger, new_best)

    logger.info("Finished training scene-level completion model.")


def validate(
    dataloader,
    model,
    logger,
    tensorboard_logger,
    cfg,
    epoch,
):
    logger.info("Evaluating scene-level completion model on validation set...")

    model.eval()

    with torch.no_grad():
        avg_L_cd1 = []
        avg_L_cd2 = []
        avg_L_cd3 = []
        avg_L_cd4 = []
        avg_L_part = []
        avg_L_center = []
        avg_L_normal = []

        for data in tqdm(dataloader, total=len(dataloader)):
            scene_id = data["scene_id"]
            partial = data["partial"][0].cuda()
            partial_normals = data["partial_normals"][0].cuda()
            gt_completions = data["complete"][0].cuda()
            gt_normals = data["complete_normals"][0].cuda()
            gt_center = data["object_center"][0].cuda()
            free_space_points = data["free_space"].cuda()
            occupied_space_points = data["occupied_space"].cuda()
            object_transforms = data["transforms"][0].cuda()

            # subsample partial point cloud
            downsampled_partials, downsampled_partial_normals = batch_downsample(
                partial, 
                n_samples=cfg.val_dataloader.input_downsample_points,
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

            # subsample ground truth point cloud
            downsampled_gt_completions = batch_downsample(
                gt_completions, 
                n_samples=cfg.val_dataloader.gt_downsample_points,
            )

            # compute completion and normal losses
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

        # log validation losses and visualization
        losses = {
            "Val L_cd1": np.mean(avg_L_cd1),
            "Val L_cd2": np.mean(avg_L_cd2),
            "Val L_cd3": np.mean(avg_L_cd3),
            "Val L_cd4": np.mean(avg_L_cd4),
            "Val L_part": np.mean(avg_L_part),
            "Val L_center": np.mean(avg_L_center),
            "Val L_normal": np.mean(avg_L_normal),
        }
        log_losses(losses, logger, tensorboard_logger, epoch, cfg.num_epochs)
        log_scene_visualizations(
            partial.detach().cpu(), 
            pred_completions[3].clone().detach().cpu(), 
            gt_completions.detach().cpu(), 
            object_transforms.detach().cpu(),
            dataloader.dataset.instance_colors,
            tensorboard_logger, 
            epoch, 
            "Validation",
        )


    return np.mean(avg_L_cd4)
