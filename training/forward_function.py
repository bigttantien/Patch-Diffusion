"""Forward function."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from diffusers import AutoencoderKL

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

#----------------------------------------------------------------------------

def forward_function(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    real_p              = 0.5,
    train_on_latents    = False,
    progressive         = False,
    device              = torch.device('cuda'),
    network_pkl         = '',
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = torch.utils.data.DataLoader(dataset=dataset_obj, sampler=None, batch_size=batch_gpu, **data_loader_kwargs)

    img_resolution, img_channels = dataset_obj.resolution, dataset_obj.num_channels

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=False) as f:
        net = pickle.load(f)['ema'].to(device)

    # methods = [method for method in dir(net) if callable(getattr(net, method))]

    # # In ra tất cả các phương thức
    # for method in methods:
    #     print(method)

    # assert 1== 9

    for batch in dataset_iterator:
        images, labels = batch
        images = images.to(device).to(torch.float32) / 127.5 - 1
        labels = labels.to(device)

        out = net(images)
        print(out.shape)

        break

    # Forward
    # dist.print0(f'Start forward for {total_kimg} kimg...')
    # dist.print0()
    # cur_nimg = resume_kimg * 1000
    # cur_tick = 0
    # tick_start_nimg = cur_nimg
    # tick_start_time = time.time()
    # maintenance_time = tick_start_time - start_time
    # dist.update_progress(cur_nimg // 1000, total_kimg)
    # stats_jsonl = None
    # batch_mul_dict = {512: 1, 256: 2, 128: 4, 64: 16, 32: 32, 16: 64}
    # if train_on_latents:
    #     p_list = np.array([(1 - real_p), real_p])
    #     patch_list = np.array([img_resolution // 2, img_resolution])
    #     batch_mul_avg = np.sum(p_list * np.array([2, 1]))
    # else:
    #     p_list = np.array([(1-real_p)*2/5, (1-real_p)*3/5, real_p])
    #     patch_list = np.array([img_resolution//4, img_resolution//2, img_resolution])
    #     batch_mul_avg = np.sum(np.array(p_list) * np.array([4, 2, 1]))  # 2

    # batch_mul = batch_mul_dict[patch_size] // batch_mul_dict[img_resolution]
    # images, labels = [], []
    # for _ in range(batch_mul):
    #     images_, labels_ = next(dataset_iterator)
    #     images.append(images_), labels.append(labels_)
    # images, labels = torch.cat(images, dim=0), torch.cat(labels, dim=0)
    # del images_, labels_
    # images = images.to(device).to(torch.float32) / 127.5 - 1

    # if train_on_latents:
    #     with torch.no_grad():
    #         images = img_vae.encode(images)['latent_dist'].sample()
    #         images = latent_scale_factor * images

    # labels = labels.to(device)
    # loss = loss_fn(net=ddp, images=images, patch_size=patch_size, resolution=img_resolution,
    #                 labels=labels, augment_pipe=augment_pipe)

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
