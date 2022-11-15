import shutil
import os
import random
from typing import List

from torch_utils import misc
import dnnlib
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as TVF

import legacy
from training.networks import Generator

def generate_videos(
    output_path: str = None,
    network_pkl: str = None,
    video_len: int = 16,
    num_videos: int = 2048,
    batch_size: int = 4,
    seed: int = 42,
    as_frames: bool = True,
    time_offset: int = 0,
    G=None,
    device=None
):
    device = torch.device('cuda')
    if device is None:
        device = torch.device('cuda')
    if network_pkl is not None:
        with dnnlib.util.open_url(network_pkl) as f:
            network = legacy.load_network_pkl(f)
            G = network['G_ema'].to(device).eval() # type: ignore
            print("Done. ")
        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device).eval()
            misc.copy_params_and_buffers(G, G2, require_all=False)
            G = G2
    else:
        assert G is not None, f"Need to specify either network pkl or G"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if as_frames:
        os.makedirs(output_path, exist_ok=True)
        curr_video_idx = 0
        all_z = torch.randn(num_videos, G.z_dim, device=device) # [num_videos, z_dim]
        all_z_motion = torch.randn_like(all_z) # [num_videos, z_motion_dim]

        for batch_idx in range((num_videos + batch_size - 1) // batch_size):
            z = all_z[batch_idx * batch_size : (batch_idx + 1) * batch_size] # [batch_size, z_dim]
            z_motion = all_z_motion[batch_idx * batch_size : (batch_idx + 1) * batch_size] # [batch_size, z_motion_dim]
            videos = lean_generation(G=G, z=z, video_len=video_len, z_motion=z_motion, noise_mode='const', time_offset=time_offset) # [b, c, t, h, w]
            videos = videos.permute(0, 2, 1, 3, 4) # [b, t, c, h, w]
            videos = (videos * 0.5 + 0.5).clamp(0, 1) # [b, t, c, h, w]

            for video in videos:
                save_video_frames_as_frames(video, os.path.join(output_path, f'{curr_video_idx:06d}'), time_offset=time_offset)
                curr_video_idx += 1

                if curr_video_idx == num_videos:
                    break


def lean_generation(G: torch.nn.Module, z: Tensor, video_len: int, frames_batch_size: int=32, time_offset: int=0, **kwargs):
    # if z_motion is None:
    #     z_motion = torch.randn(z.shape[0], 512).to(z.device) # [num_videos, z_motion_dim]
    Ts = torch.linspace(0, video_len / 16.0, steps=video_len).view(video_len, 1, 1).unsqueeze(0) + time_offset / 16.0 # [1, video_len, 1, 1]
    Ts = Ts.repeat(z.shape[0], 1, 1, 1).to(z.device) # [num_videos, video_len, 1, 1]
    all_frames = []

    for curr_batch_idx in range((video_len + frames_batch_size - 1) // frames_batch_size):
        curr_ts = Ts[:, curr_batch_idx * frames_batch_size : (curr_batch_idx + 1) * frames_batch_size, :, :] # [1, frames_batch_size, 1, 1]
        curr_ts = curr_ts.reshape(-1, 1, 1, 1) # [frames_batch_size, 1, 1, 1]
        with torch.no_grad():
            frames = G.get_final_output(z=z, c=None, timesteps=video_len, Ts=curr_ts, **kwargs).cpu() # [num_videos * frames_batch_size, c, h, w] 
        frames = frames.view(len(z), -1, *frames.shape[1:]) # [num_videos, frame_batch_size, c, h, w]
        all_frames.append(frames)

    videos = torch.cat(all_frames, dim=1) # [num_videos, video_len, c, h, w]
    videos = videos.permute(0, 2, 1, 3, 4) # [num_videos, c, video_len, h, w]

    return videos


def save_video_frames_as_frames(frames: List[Tensor], save_dir: os.PathLike, time_offset: int=0):
    os.makedirs(save_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        save_path = os.path.join(save_dir, f'{i + time_offset:06d}.jpg')
        TVF.to_pil_image(frame).save(save_path, q=95)


if __name__ == "__main__":
    generate_videos()