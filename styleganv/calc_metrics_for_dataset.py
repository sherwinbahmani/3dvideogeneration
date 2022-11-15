# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

# import sys; sys.path.extend(['.', 'src'])
import os
import shutil
import click
import tempfile
import torch
from omegaconf import OmegaConf
import random
import dnnlib
from dnnlib.util import make_cache_dir_path
from datetime import datetime

from styleganv.metrics import metric_main
from styleganv.metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from styleganv.create_fake_dataset import generate_videos

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    # sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    # training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Print network summary.
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Calculate each metric.
    results = dnnlib.EasyDict()
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(
            metric=metric,
            dataset_kwargs=args.dataset_kwargs,
            gen_dataset_kwargs=args.gen_dataset_kwargs,
            generator_as_dataset=args.generator_as_dataset,
            num_gpus=args.num_gpus,
            rank=rank,
            device=device,
            progress=progress,
            cache=args.use_cache,
            num_runs=args.num_runs,
        )
        results.update(result_dict.results)

        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir)

        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')
    return results

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

def calc_metrics_for_dataset(metrics, real_data_path, fake_data_path, resolution, mirror=None, gpus = 1, verbose = False, use_cache: bool = True, num_runs: int = 1):
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        raise ValueError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        raise ValueError('--gpus must be at least 1')

    dummy_dataset_cfg = OmegaConf.create({'max_num_frames': 10000, 'sampling': {'type': 'uniform', 'num_frames_per_video': 2, 'dists_between_frames': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                'max_dist_between_frames': 32}})

    # Initialize dataset options for real data.
    args.dataset_kwargs = dnnlib.EasyDict(
        class_name='styleganv.training.dataset.VideoFramesFolderDataset',
        path=real_data_path,
        cfg=dummy_dataset_cfg,
        xflip=mirror,
        resolution=resolution,
        use_labels=False,
    )

    # Initialize dataset options for fake data.
    args.gen_dataset_kwargs = dnnlib.EasyDict(
        class_name='styleganv.training.dataset.VideoFramesFolderDataset',
        path=fake_data_path,
        cfg=dummy_dataset_cfg,
        xflip=False,
        resolution=resolution,
        use_labels=False,
    )
    args.generator_as_dataset = True

    # Print dataset options.
    if args.verbose:
        print('Real data options:')
        print(args.dataset_kwargs)

        print('Fake data options:')
        print(args.gen_dataset_kwargs)

    # Locate run dir.
    args.run_dir = None
    args.use_cache = use_cache
    args.num_runs = num_runs

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    # torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        # if args.num_gpus == 1:
        results = subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        # else:
        #     torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)
    return results

def eval_metrics_dataset(G, metrics, real_data_path, resolution, nframes, fake_data_path=None):
    if fake_data_path is None:
        fake_data_path = make_cache_dir_path(os.path.join('metrics_dataset', datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        while os.path.exists(fake_data_path):
            fake_data_path = make_cache_dir_path(os.path.join('metrics_dataset', datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    with torch.no_grad():
        if metrics == ["fid50k_full"]:
            generate_videos(output_path=fake_data_path, G=G, batch_size=4, num_videos=50000, video_len=1)
        else:
            generate_videos(output_path=fake_data_path, G=G, video_len=nframes, batch_size=1)
        results = calc_metrics_for_dataset(
            metrics=metrics,
            real_data_path=real_data_path,
            fake_data_path=fake_data_path,
            resolution=resolution
            )
    if os.path.exists(fake_data_path):
        shutil.rmtree(fake_data_path)
    return results
    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics_for_dataset()