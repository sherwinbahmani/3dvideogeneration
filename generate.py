# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

from ast import parse
import os
import re
import time
import glob
from typing import List, Optional

from torch_utils import misc
import argparse
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
from renderer import Renderer
from training.data_utils import save_4d_videos

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def generate_sample_videos(
    outdir: str,
    network_pkl: str = None,
    seeds: Optional[List[int]] = [1],
    motion_seeds: Optional[List[int]] = None,
    truncation_psi: float = 1.0,
    noise_mode: str = "const",
    class_idx: Optional[int] = None,
    projected_w: Optional[str] = None,
    render_program=None,
    render_option=None,
    n_steps=5,
    time_steps=1,
    batch_size=1,
    relative_range_yaw_scale=1.0,
    relative_range_pitch_scale=1.0,
    fov=None,
    G=None,
    D=None,
    device=None
):

    
    if device is None:
        device = torch.device('cuda')
    if network_pkl is not None:
        if os.path.isdir(network_pkl):
            network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
        print('Loading networks from "%s"...' % network_pkl)
        
        with dnnlib.util.open_url(network_pkl) as f:
            network = legacy.load_network_pkl(f)
            G = network['G_ema'].to(device) # type: ignore
            D = network['D'].to(device)
        # from fairseq import pdb;pdb.set_trace()
    else:
        assert G is not None and D is not None, f"Need to specify either network pkl or G and D"
    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise ValueError('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # avoid persistent classes... 
    from training.networks import Generator
    # from training.stylenerf import Discriminator
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
        # D2 = Discriminator(*D.init_args, **D.init_kwargs).to(device)
        # misc.copy_params_and_buffers(D, D2, require_all=False)
    G2 = Renderer(G2, D, program=render_program)
    
    # Generate images.
    all_imgs = []

    def stack_imgs(imgs):
        img = torch.stack(imgs, dim=2)
        return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

    def proc_img(img): 
        return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    if motion_seeds is None:
        motion_seeds = seeds

    if projected_w is not None:
        ws = np.load(projected_w)
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        img = G2(styles=ws, truncation_psi=truncation_psi, noise_mode=noise_mode, render_option=render_option)
        assert isinstance(img, List)
        imgs = [proc_img(i) for i in img]
        all_imgs += [imgs]
    
    else:
        for seed_idx, seed in enumerate(seeds):
            G2.set_random_seed(seed)
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, G.z_dim)).to(device)
            relative_range_yaw = [0.5 - 0.5 * relative_range_yaw_scale, 0.5 + 0.5 * relative_range_yaw_scale]
            relative_range_pitch= [0.5 - 0.5 * relative_range_pitch_scale, 0.5 + 0.5 * relative_range_pitch_scale]
            outputs = G2(
                z=z,
                c=label,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                render_option=render_option,
                n_steps=n_steps,
                relative_range_yaw=relative_range_yaw,
                relative_range_pitch=relative_range_pitch,
                return_cameras=True,
                batch_size=batch_size,
                time_steps=time_steps,
                fov=fov,
                motion_seed=motion_seeds[seed_idx])
            if isinstance(outputs, tuple):
                img, _ = outputs
            else:
                img = outputs
            
            save_4d_videos(img, outdir, drange=[-1, 1], timesteps=time_steps, seed=seed, motion_seed=motion_seeds[seed_idx])


#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_pkl', help='Network pickle filename', required=True)
    parser.add_argument('--seeds', type=num_range, help='List of random seeds')
    parser.add_argument('--motion_seeds', type=num_range, help='List of random seeds')
    parser.add_argument('--truncation_psi', type=float, help='Truncation psi', default=1)
    parser.add_argument('--class_idx', type=int, help='Class label (unconditional if not specified)')
    parser.add_argument('--noise-mode', help='Noise mode', choices=['const', 'random', 'none'], default='const')
    parser.add_argument('--projected-w', help='Projection result file', type=str, metavar='FILE')
    parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
    parser.add_argument('--render_program', default=None)
    parser.add_argument('--render-option', default=None, type=str, help="e.g. up_256, camera, depth")
    parser.add_argument('--n_steps', default=7, type=int, help="number of steps for each seed")
    parser.add_argument('--time_steps', default=16, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--relative_range_yaw_scale', default=1.0, type=float, help="relative scale on top of the original range u")
    parser.add_argument('--relative_range_pitch_scale', default=1.0, type=float, help="relative scale on top of the original range v")
    parser.add_argument('--fov', default=None, type=float, help="fov for zooming effect")
    args = parser.parse_args()
    generate_sample_videos(**vars(args))

#----------------------------------------------------------------------------
