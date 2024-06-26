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
from tqdm import tqdm
from PIL import Image


from diffusers import AutoencoderKL

from torch_utils import persistence

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
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe

    img_resolution, img_channels = dataset_obj.resolution, dataset_obj.num_channels

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=False) as f:
        net = pickle.load(f)['ema'].to(device)
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)


    forward_fn = forward_class()

    os.mkdir(os.path.join(run_dir, "output_forward"))

    batch_name = 1
    for batch in tqdm(dataset_iterator):
        images, labels = batch
        images = images.to(device).to(torch.float32) / 127.5 - 1
        labels = labels.to(device)
        # print("Len batch:", len(images))

        x_start = 0
        y_start = 0
        image_size = img_resolution
        resolution = img_resolution
        x_pos = torch.arange(x_start, x_start+image_size).view(1, -1).repeat(image_size, 1)
        y_pos = torch.arange(y_start, y_start+image_size).view(-1, 1).repeat(1, image_size)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
        latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
        latents_pos = latents_pos.unsqueeze(0).repeat(len(images), 1, 1, 1)
        # print("latents_pos: ",latents_pos.shape)

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()
        y, augment_labels = images, None
        n = torch.randn_like(y) * sigma
        # print("sigma: ", sigma.shape)
        # print("y+n", (y+n).shape)

        test = (y+n)[0]
        output_img = Image.fromarray(((test.cpu().numpy() + 1)*127.5).astype('uint8').transpose((1, 2, 0)))
        output_img.save("test.png")
        assert 1==2

        out = ema(y + n, sigma, latents_pos, None)
        
        for i in range(len(images)):
            out_i = out[i]
            output_img = Image.fromarray(((out_i.cpu().numpy() + 1)*127.5).astype('uint8').transpose((1, 2, 0)))
            output_img_dir = os.path.join(run_dir, "output_forward", f"output_image_{i+batch_name}.png")
            output_img.save(output_img_dir)

        batch_name += 64


    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
