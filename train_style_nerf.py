import argparse
import glob
import os
import time

import numpy as np
from PIL import Image
from numpy.core.numeric import False_
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models as vmodels
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf)
from style_transfer import NST_VGG19

from nerf.volume_rendering_utils import volume_render_radiance_field
from nerf.nerf_helpers import get_minibatches, ndc_rays, sample_pdf_2 as sample_pdf
from nerf.train_utils import run_network


DEBUG_IDX = 28

# use coarse only as probe
def predict_and_render_fine_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    with torch.no_grad():
        radiance_field = run_network(
            model_coarse,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )

        (
            rgb_coarse,
            disp_coarse,
            acc_coarse,
            weights,
            depth_coarse,
        ) = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            
            render_rgb=False,
            render_acc=False,
            render_disp=False,
            render_depth=False
        )
    radiance_field = None

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )
        rgb_fine, _, _, _, _ = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,

            render_disp=False,
            render_acc=False,
            render_depth=False
        )

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, 
        help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--style-image", type=str, required=True,
        help="Path to style image file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split = load_blender_data(
                cfg.dataset.basedir,
                half_res=cfg.dataset.half_res,
                testskip=cfg.dataset.testskip,
                img_size=cfg.dataset.img_size
            )
            i_train, i_val, i_test = i_split
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]

            mask = images[..., -1:]
            images = images[..., :3]
            if cfg.nerf.train.white_background:
                images = images * mask + (1.0 - mask)
            patch_size = int(H * cfg.dataset.patch_ratio)

        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :3, :4]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array(
                [
                    i
                    for i in np.arange(images.shape[0])
                    if (i not in i_test and i not in i_val)
                ]
            )
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)

            patch_size = cfg.dataset.patch_size
        
        # square patch for now (patch_w == patch_h)
        p_w = p_h = patch_size
        print('image size (H, W):', H, W)
        print('patch size (p_h, p_w):', p_h, p_w)

        # w_offset = patch_size // 2
        # h_offset = patch_size // 2

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)
    # Initialize a fine model.
    model_fine = getattr(models, cfg.models.fine.type)(
        num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
        include_input_xyz=cfg.models.fine.include_input_xyz,
        include_input_dir=cfg.models.fine.include_input_dir,
        use_viewdirs=cfg.models.fine.use_viewdirs,
    )
    model_fine.to(device)
    # Load style image & Initialize VGG model
    assert os.path.exists(configargs.style_image)
    style_img = transforms.ToTensor()(Image.open(configargs.style_image))[None]
    style_img = F.interpolate(style_img, size=(224, 224), mode='bilinear')
    nst_vgg19 = NST_VGG19([style_img],
                          cfg.models.style.content_layers,
                          cfg.models.style.style_layers)
    nst_vgg19 = nst_vgg19.to(device)
    # Move loss targets to device manually 
    for sl in nst_vgg19.style_losses:
      sl._target = sl._target.to(device)
    for cl in nst_vgg19.content_losses:
      cl._target = cl._target.to(device)
    style_img = style_img.to(device)

    # No grad required for vgg model
    nst_vgg19.requires_grad_(False)

    # Initialize optimizer.
    trainable_parameters = []
    for name, param in model_fine.named_parameters():
        if name.split('.')[0] in {'fc_feat', 'layers_dir', 'fc_rgb'}:
            trainable_parameters.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0
    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # start_iter = checkpoint["iter"]

    # # TODO: Prepare raybatch tensor if batching random rays

    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.eval()
        model_fine.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
        else:
            img_idx = np.random.choice(i_train)
            # img_idx = DEBUG_IDX
            img_target = images[img_idx].to(device)
            pose_target = poses[img_idx, :3, :4].to(device)
            # mask_target = mask[img_idx].to(device)
            
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            viewdirs = ray_directions
            viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
            viewdirs = viewdirs.view((-1, 3))

            # if cfg.dataset.no_ndc is False:
            if False:
                ro, rd = ndc_rays(H, W, focal, 1.0, ray_origins, ray_directions)
                ro = ro.view((-1, 3))
                rd = rd.view((-1, 3))
            else:
                ro = ray_origins.view((-1, 3))
                rd = ray_directions.view((-1, 3))

            near = cfg.dataset.near * torch.ones_like(rd[..., :1])
            far = cfg.dataset.far * torch.ones_like(rd[..., :1])
            rays = torch.cat((ro, rd, near, far), dim=-1)
            if cfg.nerf.use_viewdirs:
                rays = torch.cat((rays, viewdirs), dim=-1)

            # sample random patch
            if (p_h != H or p_w != W):
                tl_h = np.random.randint(0, H-p_h+1)
                tl_w = np.random.randint(0, W-p_w+1)
                inds = torch.arange(W*H).reshape(H, W)
                select_inds = inds[tl_h:tl_h+p_h,
                                   tl_w:tl_w+p_w].flatten()
                rays = rays[select_inds]
                target_rgb = img_target[tl_h:tl_h+p_h,
                                        tl_w:tl_w+p_w]
                print(rays.shape, target_rgb.shape)
            else:
                target_rgb = img_target
            
            # predict and render
            then = time.time()
            rays = rays.to(device)

            _, _, _, rgb_fine, _, _ = predict_and_render_fine_radiance(
                rays,
                model_coarse,
                model_fine,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )

        rgb_fine = rgb_fine.reshape_as(target_rgb)
        rgb_fine = F.interpolate(
            rgb_fine.movedim(-1, 0)[None], 
            size=(224, 224), mode='bilinear')

        target_rgb = F.interpolate(
            target_rgb.movedim(-1, 0)[None],
            size=(224, 224), mode='bilinear')

        optimizer.zero_grad()
        nst_vgg19.update_content(target_rgb)

        content_loss = 0
        style_loss = 0
        nst_vgg19(rgb_fine)  # forward pass
        for cl in nst_vgg19.content_losses:
            content_loss += cfg.models.style.content_weight * cl.loss

        # # set style image as background
        # mask_target = mask_target.movedim(-1, 0)[None]
        # rgb_fine = rgb_fine * mask_target + style_img * (1 - mask_target)
        # nst_vgg19(rgb_fine)  # forward pass

        for sl in nst_vgg19.style_losses:
            style_loss += cfg.models.style.style_weight * sl.loss
        loss = content_loss + style_loss 
        loss.backward()
        optimizer.step()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                f"[TRAIN] Iter: {i} "
                f"Loss: {loss.item()} "
                f"= {content_loss.item()} + {style_loss.item()}"
            )
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/content_loss", content_loss.item(), i)
        writer.add_scalar("train/style_loss", style_loss.item(), i)

        if i == 0:
            rgb_fine = rgb_fine.cpu().detach()
            rgb_fine = rgb_fine[0].movedim(0, -1)
            rgb_fine = (rgb_fine.numpy() * 255).astype(np.uint8)
            im = Image.fromarray(rgb_fine)
            im.save('/content/hi.png')
            print('saved at /content/hi.png')

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            model_fine.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    # img_idx = DEBUG_IDX
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H, W, focal, pose_target
                    )
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                    loss = fine_loss
                else:
                    loss = coarse_loss
                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


if __name__ == "__main__":
    main()
