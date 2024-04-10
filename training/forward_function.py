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
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    img_resolution, img_channels = dataset_obj.resolution, dataset_obj.num_channels

    if train_on_latents:
        # img_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae").to(device)
        img_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
        img_vae.eval()
        set_requires_grad(img_vae, False)
        latent_scale_factor = 0.18215
        img_resolution, img_channels = dataset_obj.resolution // 8, 4
    else:
        img_vae = None

    # # Construct network.
    # dist.print0('Constructing network...')
    # net_input_channels = img_channels + 2
    # interface_kwargs = dict(img_resolution=img_resolution,
    #                         img_channels=net_input_channels,
    #                         out_channels=4 if train_on_latents else dataset_obj.num_channels,
    #                         label_dim=dataset_obj.label_dim)
    # net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    # net.train().requires_grad_(True).to(device)
    
    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=False) as f:
        net = pickle.load(f)['ema'].to(device)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            x_pos = torch.zeros([batch_gpu, 2, net.img_resolution, net.img_resolution], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, x_pos, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    print(len(dataset_iterator))
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
