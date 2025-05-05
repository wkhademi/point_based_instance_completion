import torch
import numpy as np


def log_model_parameters(model, logger):
    """
    Log model layers and their parameter counts.
    """
    logger.info("----------------------- Model Layers and Parameter Count -----------------------")
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                param_count = name + ": " + "x".join(str(x) for x in list(param.size())) + " = " + str(num_param)
                logger.info(param_count)
            else:
                logger.info("%s: %d"%(name, num_param))
            total_param += num_param

    logger.info("Number of trainable parameters: %d"%total_param)


def log_visualizations(
    partials, 
    preds, 
    gts,
    tensorboard_logger, 
    epoch, 
    mode,
):
    # grab first example in batch to visualize
    partial = partials[0][0:1].detach().cpu()
    P1 = preds[0][0:1].detach().cpu()
    P2 = preds[1][0:1].detach().cpu()
    P3 = preds[2][0:1].detach().cpu()
    P4 = preds[3][0:1].detach().cpu()
    gt = gts[0:1].detach().cpu()

    # visualize point clouds in tensorboard
    tensorboard_logger.add_mesh(f"{mode} Partial", vertices=partial, global_step=epoch)
    tensorboard_logger.add_mesh(f"{mode} P1", vertices=P1, global_step=epoch)
    tensorboard_logger.add_mesh(f"{mode} P2", vertices=P2, global_step=epoch)
    tensorboard_logger.add_mesh(f"{mode} P3", vertices=P3, global_step=epoch)
    tensorboard_logger.add_mesh(f"{mode} P4", vertices=P4, global_step=epoch)
    tensorboard_logger.add_mesh(f"{mode} Ground Truth", vertices=gt, global_step=epoch)
    tensorboard_logger.flush()


def log_scene_visualizations(
    partials, 
    preds, 
    gts,
    object_transforms,
    instance_colors,
    tensorboard_logger, 
    epoch, 
    mode,
):
    # color objects by instance
    partial_colors = []
    complete_colors = []
    for idx in range(partials.shape[0]):
        partial_instance_ids = np.full((partials[idx].shape[0],), idx)
        complete_instance_ids = np.full((gts[idx].shape[0],), idx)
        partial_colors.append(instance_colors[partial_instance_ids])
        complete_colors.append(instance_colors[complete_instance_ids])
    partial_colors = torch.tensor(np.stack(partial_colors, axis=0).reshape(1, -1, 3))
    complete_colors = torch.tensor(np.stack(complete_colors, axis=0).reshape(1, -1, 3))

    # transform object instances back to the scene coordinate system
    partials = ((partials * object_transforms[..., 3:]) + object_transforms[..., :3]).reshape(1, -1, 3)
    preds = ((preds * object_transforms[..., 3:]) + object_transforms[..., :3]).reshape(1, -1, 3)
    gts = ((gts * object_transforms[..., 3:]) + object_transforms[..., :3]).reshape(1, -1, 3)

    # visualize point clouds in tensorboard
    tensorboard_logger.add_mesh(f"{mode} Partial", vertices=partials, colors=partial_colors, global_step=epoch)
    tensorboard_logger.add_mesh(f"{mode} Completion", vertices=preds, colors=complete_colors, global_step=epoch)
    tensorboard_logger.add_mesh(f"{mode} Ground Truth", vertices=gts, colors=complete_colors, global_step=epoch)
    tensorboard_logger.flush()


def log_losses(losses, logger, tensorboard_logger, epoch, max_epoch):
    logger.info(f"---------- Epoch {str(epoch)}/{str(max_epoch)} ----------")
    for name, loss in losses.items():
        tensorboard_logger.add_scalar(name, loss, global_step=epoch)
        logger.info(f"{name}: {str(loss)}")
    tensorboard_logger.flush()
