import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import io
from tqdm import tqdm
from datetime import datetime
import random
import imageio
from pathlib import Path
from distillation_utils import (
            get_t_schedule,
            get_loss_weights,
            sds_vsd_grad_diffuser,
            phi_vsd_grad_diffuser,
            extract_lora_diffusers,
            predict_noise0_diffuser,
            get_images,
            get_latents,
            setup_logger
            )
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler
import hydra
from omegaconf import OmegaConf
from utils import SLURM_OUTPUT_DIR

@hydra.main(config_path="conf/distillation",
            config_name="config", version_base=None)
def main(cfg):
    # for sds and t2i, use only cfg.batch_size
    if cfg.generation_mode in ['t2i', 'sds']:
        cfg.particle_num_vsd = cfg.batch_size
        cfg.particle_num_phi = cfg.batch_size
    assert (cfg.batch_size >= cfg.particle_num_vsd) and (cfg.batch_size >= cfg.particle_num_phi)
    if cfg.batch_size > cfg.particle_num_vsd:
        print(f'use multiple ({cfg.batch_size}) particles!! Will get inconsistent x0 recorded')
    ### set random seed everywhere
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)  # for multi-GPU.
    np.random.seed(cfg.seed)  # Numpy module.
    random.seed(cfg.seed)  # Python random module.
    torch.manual_seed(cfg.seed)

    if os.getenv("SLURM_JOB_ID"):
        output_dir = os.path.join("out", SLURM_OUTPUT_DIR)
    else:
        output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,"cfg.yaml"), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 # use float32 by default
    image_name = cfg.prompt.replace(' ', '_')

    logger = setup_logger(output_dir)
    logger.info(f'[INFO] Cmdline: '+' '.join(sys.argv))
    logger.info(f'Using device: {device}; version: {str(torch.version.cuda)}')
    if device.type == 'cuda':
        logger.info(torch.cuda.get_device_name(0))

    #######################################################################################
    ### load model
    vae = AutoencoderKL.from_pretrained(cfg.model_id, subfolder="vae",
            cache_dir=cfg.model_dir, torch_dtype=dtype, local_files_only=cfg.local_files_only)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id, subfolder="tokenizer",
            cache_dir=cfg.model_dir, torch_dtype=dtype, local_files_only=cfg.local_files_only)
    text_encoder = CLIPTextModel.from_pretrained(cfg.model_id, subfolder="text_encoder", cache_dir=cfg.model_dir, torch_dtype=dtype, local_files_only=cfg.local_files_only)
    unet = UNet2DConditionModel.from_pretrained(cfg.model_id, subfolder="unet", cache_dir=cfg.model_dir, torch_dtype=dtype, local_files_only=cfg.local_files_only)
    scheduler = DDIMScheduler.from_pretrained(
           cfg.model_id, subfolder="scheduler",
           cache_dir=cfg.model_dir, torch_dtype=dtype, local_files_only=cfg.local_files_only)

    if cfg.half_inference:
        unet = unet.half()
        vae = vae.half()
        text_encoder = text_encoder.half()
    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    if cfg.generation_mode == 'sds':
        unet_phi = None
        vae_phi = vae

    ### get text embedding
    text_input = tokenizer([cfg.prompt] * max(cfg.particle_num_vsd,cfg.particle_num_phi), padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * max(cfg.particle_num_vsd,cfg.particle_num_phi), padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings_vsd = torch.cat([uncond_embeddings[:cfg.particle_num_vsd], text_embeddings[:cfg.particle_num_vsd]])
    text_embeddings_phi = torch.cat([uncond_embeddings[:cfg.particle_num_phi], text_embeddings[:cfg.particle_num_phi]])

    ### weight loss
    num_train_timesteps = len(scheduler.betas)
    loss_weights = get_loss_weights(scheduler.betas, cfg)
    scheduler.set_timesteps(num_train_timesteps)

    ### initialize particles
    if cfg.rgb_as_latents:
        particles = torch.randn((cfg.batch_size, unet.config.in_channels,
             cfg.height // 8, cfg.width // 8))
    else:
        particles = torch.randn((cfg.batch_size, 3, cfg.height, cfg.width))
        cfg.lr = cfg.lr * 1   # need larger lr for rgb particles
    particles = particles.to(device, dtype=dtype)
    particles.requires_grad = True
    particles_to_optimize = [particles]

    total_parameters = sum(p.numel() for p in particles_to_optimize if p.requires_grad)
    print(f'Total number of trainable parameters in particles: {total_parameters}; number of particles: {cfg.batch_size}')
    optimizer = torch.optim.Adam(particles_to_optimize, lr=cfg.lr)

    #######################################################################################
    ############################# Main optimization loop ##############################
    log_steps = []
    train_loss_values = []
    ave_train_loss_values = []
    if cfg.log_progress:
        image_progress = []
    first_iteration = True
    logger.info("################# Metrics: ####################")
    ######## t schedule #########
    chosen_ts = get_t_schedule(num_train_timesteps, cfg, loss_weights)
    pbar = tqdm(chosen_ts)

    if cfg.generation_mode in ['sds', 'vsd']:
        cross_attention_kwargs = {}
        for step, chosen_t in enumerate(pbar):
            # get latent of all particles
            latents = get_latents(particles, vae, cfg.rgb_as_latents)
            t = torch.tensor([chosen_t]).to(device)
            ######## q sample #########
            # random sample particle_num_vsd particles from latents
            indices = torch.randperm(latents.size(0))
            latents_vsd = latents[indices[:cfg.particle_num_vsd]]
            noise = torch.randn_like(latents_vsd)
            noisy_latents = scheduler.add_noise(latents_vsd, noise, t)
            ######## Do the gradient for latents!!! #########
            optimizer.zero_grad()
            grad_, noise_pred, noise_pred_phi = sds_vsd_grad_diffuser(unet, noisy_latents, noise, text_embeddings_vsd, t, \
                                                    guidance_scale=cfg.guidance_scale, unet_phi=unet_phi, \
                                                        generation_mode=cfg.generation_mode, phi_model=cfg.phi_model, \
                                                            cross_attention_kwargs=cross_attention_kwargs, \
                                                                multisteps=cfg.multisteps, scheduler=scheduler, lora_v=cfg.lora_vprediction, \
                                                                    half_inference=cfg.half_inference, \
                                                                        cfg_phi=cfg.cfg_phi, grad_scale=cfg.grad_scale)
            ## weighting
            grad_ *= loss_weights[int(t)]
            target = (latents_vsd - grad_).detach()
            loss = 0.5 * F.mse_loss(latents_vsd, target, reduction="mean") / cfg.batch_size
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            train_loss_values.append(loss.item())
            pbar.set_description(f'Loss: {loss.item():.6f}, sampled t : {t.item()}')

            optimizer.zero_grad()

            ######## Evaluation and log metric #########
            if cfg.log_steps and (step % cfg.log_steps == 0 or step == (cfg.num_steps-1)):
                log_steps.append(step)
                # save current img_tensor
                # scale and decode the image latents with vae
                tmp_latents = 1 / vae.config.scaling_factor * latents_vsd.clone().detach()
                if cfg.save_x0:
                    # compute the predicted clean sample x_0
                    pred_latents = scheduler.step(noise_pred, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                with torch.no_grad():
                    if cfg.half_inference:
                        tmp_latents = tmp_latents.half()
                    image_ = vae.decode(tmp_latents).sample.to(torch.float32)
                    if cfg.save_x0:
                        if cfg.half_inference:
                            pred_latents = pred_latents.half()
                        image_x0 = vae.decode(pred_latents / vae.config.scaling_factor).sample.to(torch.float32)
                        image = torch.cat((image_,image_x0), dim=2)
                    else:
                        image = image_
                if cfg.log_progress:
                    image_progress.append((image/2+0.5).clamp(0, 1))
                save_image((image/2+0.5).clamp(0, 1), f'{output_dir}/step{step}_t{t.item()}.png')

    if cfg.log_gif:
        images = sorted(Path(output_dir).glob(f"*{image_name}*.png"))
        images = [imageio.imread(image) for image in images]
        imageio.mimsave(f'{output_dir}/train_progress.gif', images, duration=0.3)
    if cfg.log_progress and cfg.batch_size == 1:
        concatenated_images = torch.cat(image_progress, dim=0)
        save_image(concatenated_images, f'{output_dir}/{image_name}_prgressive.png')

    image = get_images(particles, vae, cfg.rgb_as_latents)
    save_image((image/2+0.5).clamp(0, 1), f'{output_dir}/final_image.png')

#########################################################################################
if __name__ == "__main__":
    main()
