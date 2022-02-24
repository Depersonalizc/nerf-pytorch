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
DEBUG_P_IDX = 0

INCRE_N = 2  # no. views to incrementally learn

# use coarse only as probe
def predict_and_render_fine_radiance(
    ray_batch,
    model_coarse,
    model_addon, # fine + new appearance
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
            model_addon,
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
        "--base-checkpoint",
        type=str,
        default="",
        help="Path to base NeRF checkpoint.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to base+addon checkpoint.",
    )
    parser.add_argument(
        "--reset-iter",
        action='store_true',
        help="Start iter from 0 instead of loading from ckpt.",
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

        # load all proxy images
        if hasattr(cfg.dataset, "proxydir") and os.path.exists(cfg.dataset.proxydir):
            print(f'loading proxy images from {cfg.dataset.proxydir}...')
            imgdirs = sorted(
                [os.path.join(cfg.dataset.proxydir, name)
                for name in os.listdir(cfg.dataset.proxydir)
                if name[-4:].lower() in {'.png', '.jpg', '.jpeg'}]
            )
            proxy_imgs = [
                torch.from_numpy(
                    np.array(Image.open(p), 
                    dtype=np.float32)
                ) / 255
                for p in imgdirs
            ]
        else:
            print('Init proxy images as content images...')
            proxy_imgs = images

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
    model_coarse.requires_grad_(False)

    # Initialize a fine model.
    model_fine = getattr(models, cfg.models.fine.type)(
        num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
        include_input_xyz=cfg.models.fine.include_input_xyz,
        include_input_dir=cfg.models.fine.include_input_dir,
        use_viewdirs=cfg.models.fine.use_viewdirs,
    )
    model_fine.to(device)
    model_fine.requires_grad_(False)

    # Initialize an appearance model
    model_app = models.AppearanceNeRFModel(
        num_layers=7,
        hidden_size=256,
        skip_connect_every=3,
        num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
        include_input_xyz=cfg.models.fine.include_input_xyz,
    )
    model_app.to(device)
    model_app.requires_grad_()

    model_addon = models.AddonNeRFModel(model_fine, model_app)

    # Initialize VGG model w/ style image
    assert os.path.exists(configargs.style_image)
    style_img = transforms.ToTensor()(Image.open(configargs.style_image))[None]
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

    # Disable grad update for VGG model
    nst_vgg19.requires_grad_(False)

    # Filter proxy images (train set only)
    proxy_imgs = [
        proxy_imgs[i].to(device).requires_grad_()
        for i in i_train
    ]

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
        print(f'loading base+addon NeRF from {configargs.load_checkpoint}...')
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        model_app.load_state_dict(checkpoint["model_app_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if not configargs.reset_iter:
            start_iter = checkpoint["iter"] + 1
    elif os.path.exists(configargs.base_checkpoint):
        print(f'loading base NeRF from {configargs.base_checkpoint}...')
        checkpoint = torch.load(configargs.base_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
    else:
        print(f'No base nor addon NeRF ckpt found at {configargs.base_checkpoint}, starting anew...')

    # # TODO: Prepare raybatch tensor if batching random rays


    # TRAINING LOOP
    # for i in trange(start_iter, cfg.experiment.train_iters):
    coords = torch.stack(
        meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
        dim=-1,
    )
    coords = coords.reshape((-1, 2))

    all_ray_origins = []
    all_ray_directions = []
    i = -1
    for p_idx in range(INCRE_N):
        img_idx = i_train[p_idx]
        proxy_target = proxy_imgs[p_idx]  # (H, W, 3)
        img_target = images[img_idx].to(device)  # ContentLoss
        pose_target = poses[img_idx, :3, :4].to(device)  # rays_joint
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)  # rays_joint
        rays_per_view = cfg.nerf.train.num_rays_fixed // max(p_idx, 1)  # rays_fixed

        nst_vgg19.update_content(img_target.movedim(-1, 0)[None])

        # Initialize optimizer.
        optimizer = getattr(torch.optim, cfg.optimizer.type)([
            {'params': model_app.parameters(), 'lr': cfg.optimizer.lr},
            {'params': [proxy_target], 'lr': cfg.optimizer.lr_proxy}
        ])
        print(f'Jointly trainig proxy: {p_idx}, {cfg.nerf.train.num_rays_joint} rays')
        print(f'Fixed proxy: {list(range(p_idx))}, {rays_per_view} rays/view')

        # Training loop of added proxy
        for i_p in trange(
            # cfg.experiment.train_iters_per_view
            {0: 1000,
             1: 5000,
             2: 8000,
             3: 12000}[p_idx]
        ):

            # Global iteration
            i += 1
            # i = p_idx * cfg.experiment.train_iters_per_view + i_p

            def sample_rays(
                coords, size, ray_origins, ray_directions, 
                img_target, no_ndc, near, far, use_viewdirs,
            ):
                # rays: (size, rays_dim)
                select_inds = np.random.choice(coords.shape[0], size=(size), replace=False)
                select_inds = coords[select_inds]
                ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
                ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
                viewdirs = ray_directions
                viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
                viewdirs = viewdirs.view((-1, 3))
                if no_ndc is False:
                    ro, rd = ndc_rays(H, W, focal, 1.0, ray_origins, ray_directions)
                    ro = ro.view((-1, 3))
                    rd = rd.view((-1, 3))
                else:
                    ro = ray_origins.view((-1, 3))
                    rd = ray_directions.view((-1, 3))
                near = cfg.dataset.near * torch.ones_like(rd[..., :1])
                far = cfg.dataset.far * torch.ones_like(rd[..., :1])
                rays = torch.cat((ro, rd, near, far), dim=-1)
                if use_viewdirs:
                    rays = torch.cat((rays, viewdirs), dim=-1)

                # target_rgb: (size, 3)
                target_rgb = img_target[select_inds[:, 0], select_inds[:, 1], :]

                return rays, target_rgb

            # Sample rays_joint
            rays_joint, target_rgb_joint = sample_rays(
                coords, cfg.nerf.train.num_rays_joint, ray_origins, ray_directions, proxy_target, 
                cfg.dataset.no_ndc, cfg.dataset.near, cfg.dataset.far, cfg.models.fine.use_viewdirs
            )

            # Sample rays_fixed
            if p_idx == 0:
                rays_fixed, target_rgb_fixed = sample_rays(
                    coords, cfg.nerf.train.num_rays_fixed, ray_origins, ray_directions, proxy_target, 
                    cfg.dataset.no_ndc, cfg.dataset.near, cfg.dataset.far, cfg.models.fine.use_viewdirs
                )
            else:
                rays_fixed, target_rgb_fixed = zip(*[
                    sample_rays(
                        coords, rays_per_view, ray_origins, ray_directions, proxy_img, 
                        cfg.dataset.no_ndc, cfg.dataset.near, cfg.dataset.far, cfg.models.fine.use_viewdirs
                    )
                    for ray_origins, ray_directions, proxy_img
                    in zip(all_ray_origins, all_ray_directions, proxy_imgs)
                ])
                rays_fixed = torch.cat(rays_fixed, 0)
                target_rgb_fixed = torch.cat(target_rgb_fixed, 0)

            rays_joint = rays_joint.to(device)
            rays_fixed = rays_fixed.to(device)

            # print(rays_fixed.shape, target_rgb_fixed.shape)
            # assert False

            # Predict and render
            then = time.time()

            _, _, _, rgb_fine_fixed, _, _ = predict_and_render_fine_radiance(
                rays_fixed,
                model_coarse,
                model_addon,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )

            _, _, _, rgb_fine_joint, _, _ = predict_and_render_fine_radiance(
                rays_joint,
                model_coarse,
                model_addon,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )


            # Sim Loss (fixed)
            sim_loss_fixed = F.mse_loss(rgb_fine_fixed, target_rgb_fixed) 
            sim_loss_fixed = sim_loss_fixed * cfg.models.style.sim_fixed_weight

            # Sim Loss (joint)
            sim_loss_joint = F.mse_loss(rgb_fine_joint, target_rgb_joint) 
            sim_loss_joint = sim_loss_joint * cfg.models.style.sim_joint_weight

            # Content & Style Loss
            with torch.no_grad():
                proxy_target.clamp_(0, 1)  # range correction
            nst_vgg19(proxy_target.movedim(-1, 0)[None])  # forward pass

            content_loss = 0
            style_loss = 0
            for cl in nst_vgg19.content_losses:
                content_loss += cfg.models.style.content_weight * cl.loss
            for sl in nst_vgg19.style_losses:
                style_loss += cfg.models.style.style_weight * sl.loss

            loss = content_loss + style_loss + sim_loss_fixed + sim_loss_joint

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Learning rate updates
            # num_decay_steps = cfg.scheduler.lr_decay * 1000
            num_decay_steps = cfg.experiment.train_iters_per_view * 0.8
            lr_new = cfg.optimizer.lr * (
                cfg.scheduler.lr_decay_factor ** (
                    min(i_p / num_decay_steps, 1.0)
                )
            )
            optimizer.param_groups[0]['lr'] = lr_new

            # Logging
            if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
                tqdm.write(
                    f"[TRAIN] Iter: {i}  "
                    f"Loss: {loss.item():.4f} "
                    f"= (C){content_loss.item():.4f} + (S){style_loss.item():.4f} "
                    f"+ (Sj){sim_loss_joint.item():.4f} + (Sf){sim_loss_fixed.item():.4f}"
                )

            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/content_loss", content_loss.item(), i)
            writer.add_scalar("train/style_loss", style_loss.item(), i)
            writer.add_scalar("train/sim_loss_joint", sim_loss_joint.item(), i)
            writer.add_scalar("train/sim_loss_fixed", sim_loss_fixed.item(), i)


            # if i == 0:
            #     im = Image.fromarray(
            #         (proxy_target.cpu().detach().numpy() * 255).astype(np.uint8)
            #     )
            #     im.save(f'/content/proxy_{p_idx}.png')
            #     print(f'saved at /content/proxy_{p_idx}.png')
            # assert False
            # Validation
            if (
                i % cfg.experiment.validate_every == 0
                or i == cfg.experiment.train_iters - 1
            ):
                tqdm.write(f"[VAL] =======> Iter: {i}")

                start = time.time()
                with torch.no_grad():
                    rgb_coarse, rgb_fine = None, None
                    target_ray_values = None

                    p_idx_val = DEBUG_P_IDX
                    img_idx_val = i_train[p_idx_val]

                    # img_idx_val = np.random.choice(i_val)
                    # img_idx_val = DEBUG_IDX
                    img_target_val = images[img_idx_val].to(device)
                    pose_target_val = poses[img_idx_val, :3, :4].to(device)
                    ray_origins_val, ray_directions_val = get_ray_bundle(
                        H, W, focal, pose_target_val
                    )
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_addon,
                        ray_origins_val,
                        ray_directions_val,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target_val

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

                    # p_idx = DEBUG_P_IDX
                    # proxy_target = proxy_imgs[p_idx]  # (H, W, 3)
                    # img_idx = i_train[p_idx]
                    # writer.add_image(
                    #     f"proxy/{img_idx}",
                    #     cast_to_image(proxy_target[..., :3].clamp(0, 1)),
                    #     i,
                    # )

                    for p_idx_, proxy_target_ in enumerate(
                      # proxy_imgs[:4]
                      proxy_imgs[:INCRE_N]
                      ):
                        img_idx = i_train[p_idx_]
                        writer.add_image(
                            f"proxy/{img_idx}",
                            cast_to_image(proxy_target_[..., :3].clamp(0, 1)),
                            i,
                        )

                    # writer.add_image(
                    #     f"proxy/{img_idx}",
                    #     cast_to_image(proxy_target[..., :3].clamp(0, 1)),
                    #     i,
                    # )                    


            # Save model
            if (i % cfg.experiment.save_every == 0 or 
                i == cfg.experiment.train_iters - 1):
                checkpoint_dict = {
                    "iter": i,
                    "model_coarse_state_dict": model_coarse.state_dict(),
                    "model_fine_state_dict": None
                    if not model_fine
                    else model_fine.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "psnr": psnr,
                    "model_app_state_dict": model_app.state_dict(),
                }
                torch.save(
                    checkpoint_dict,
                    os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
                )
                tqdm.write("================== Saved Checkpoint =================")

        # Add rays to prev list; Freeze last proxy.
        all_ray_origins.append(ray_origins)
        all_ray_directions.append(ray_directions)
        proxy_target.requires_grad_(False)

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
