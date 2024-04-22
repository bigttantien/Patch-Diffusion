# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib

#----------------------------------------------------------------------------

def generate_image_grid(
    network_pkl, dest_path,
    seed=0, gridw=8, gridh=8, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=False) as f:
        net = pickle.load(f)['ema'].to(device)

    # Pick latents and labels.
    print(f'Generating {batch_size} images...')
    latents = torch.randn([batch_size, 3, net.img_resolution, net.img_resolution], device=device)
    print("latents shape: ", latents.shape)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_start = 0
    y_start = 0
    image_size = net.img_resolution
    resolution = net.img_resolution
    x_pos = torch.arange(x_start, x_start+image_size).view(1, -1).repeat(image_size, 1)
    y_pos = torch.arange(y_start, y_start+image_size).view(-1, 1).repeat(1, image_size)
    x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
    y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
    latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
    latents_pos = latents_pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    print("x_next shape: ", x_next.shape)
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        # print("x_hat:", x_hat.shape, "t_hat:", t_hat.shape)
        denoised = net(x_hat, t_hat, latents_pos, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, latents_pos, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    # Save image grid.
    print(f'Saving image grid to "{dest_path}"...')
    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    print(image.shape)
    image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, 3)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(dest_path)
    print('Done.')

#----------------------------------------------------------------------------

def main():
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    # generate_image_grid(f'{model_root}/edm-cifar10-32x32-cond-vp.pkl',   'cifar10-32x32.png',  num_steps=18) # FID = 1.79, NFE = 35
    # generate_image_grid(f'{model_root}/edm-ffhq-64x64-uncond-vp.pkl',    'ffhq-64x64.png',     num_steps=40) # FID = 1.97, NFE = 79
    # generate_image_grid(f'{model_root}/edm-afhqv2-64x64-uncond-vp.pkl',  'afhqv2-64x64.png',   num_steps=40) # FID = 1.96, NFE = 79
    generate_image_grid("training-runs/00011-cashew_normal-64x64-uncond-ddpmpp-pedm-gpus1-batch64-fp32/network-snapshot-000363.pkl",  'cashew-64x64.png',   num_steps=100) # FID = 1.96, NFE = 79
    # generate_image_grid(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', 'imagenet-64x64.png', num_steps=256, S_churn=40, S_min=0.05, S_max=50, S_noise=1.003) # FID = 1.36, NFE = 511

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
