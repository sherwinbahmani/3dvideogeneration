# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import PIL.Image
import torch
import cv2, albumentations
import numpy as np
import torchvision
from einops import rearrange
import math
import os


def save_image(img, filename):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(filename)

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def resize_image(img_pytorch, curr_res):
    img = img_pytorch.permute(0,2,3,1).cpu().numpy()
    img = [albumentations.geometric.functional.resize(
        img[i], height=curr_res, width=curr_res, interpolation=cv2.INTER_LANCZOS4)
        for i in range(img.shape[0])]
    img = torch.from_numpy(np.stack(img, axis=0)).permute(0,3,1,2).to(img_pytorch.device)
    return img
    
def save_4d_videos(imgs, outdir, drange, timesteps, seed, normalize=True, motion_seed=None):
    imgs = imgs[::-1] # change rotation angle direction
    os.makedirs(outdir, exist_ok=True)
    motion_seed_str = '' if motion_seed == seed else f'_motion_seed_{motion_seed}'
    imgs_conv = []
    for n_step, img in enumerate(imgs):
        img = rearrange(img, '(b t) c h w -> b c t h w', t=timesteps).cpu().numpy()
        if normalize:
            lo, hi = drange
            img = np.asarray(img, dtype=np.float32)
            img = (img - lo) * (255 / (hi - lo))
            img = np.rint(img).clip(0, 255).astype(np.uint8)
        imgs_conv.append(np.copy(img))
    
    # (n_steps, batch_size, C, T, H, W)
    imgs_conv = np.stack(imgs_conv)

    # Only first frame
    imgs_start = []
    imgs_dynamic = []
    n_steps = imgs_conv.shape[0]
    for i in range(n_steps):
        imgs_start.append(imgs_conv[i, 0, :, 0])
        imgs_dynamic.append(imgs_conv[i, 0, :, i])
    
    imgs_start_stack = np.stack(imgs_start).transpose(0, 2, 3, 1)
    imgs_dynamic_stack = np.stack(imgs_dynamic).transpose(0, 2, 3, 1)
    # torchvision.io.write_video(os.path.join(outdir, f'seed_{seed}{motion_seed_str}_steps_{n_step}_static.mp4'), torch.from_numpy(imgs_start_stack), fps=8)
    torchvision.io.write_video(os.path.join(outdir, f'seed_{seed}{motion_seed_str}_steps_{n_step}.mp4'), torch.from_numpy(imgs_dynamic_stack), fps=8)