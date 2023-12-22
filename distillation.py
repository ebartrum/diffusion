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
import imageio
from pathlib import Path
from utils import (
            get_t_schedule,
            get_loss_weights,
            predict_noise,
            get_outputs,
            setup_logger,
            show_step
            )
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from utils import SLURM_OUTPUT_DIR, seed_all
import subprocess

@hydra.main(config_path="conf/distillation",
            config_name="config", version_base=None)
def main(cfg):
    seed_all(cfg.seed)
    if os.getenv("SLURM_JOB_ID"):
        output_dir = os.path.join("out", SLURM_OUTPUT_DIR)
    else:
        output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,"cfg.yaml"), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    logger = setup_logger(output_dir)
    logger.info('git commit: ' + subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip())
    logger.info('python: '+' '.join(sys.argv))
    if device.type == 'cuda':
        logger.info('GPU: ' + torch.cuda.get_device_name(0))

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

    text_input1 = tokenizer([cfg.prompt1], padding="max_length",
           max_length=tokenizer.model_max_length, truncation=True,
           return_tensors="pt")
    text_input2 = tokenizer([cfg.prompt2], padding="max_length",
           max_length=tokenizer.model_max_length, truncation=True,
           return_tensors="pt")

    with torch.no_grad():
        text_embeddings1 = text_encoder(text_input1.input_ids.to(device))[0]
        text_embeddings2 = text_encoder(text_input2.input_ids.to(device))[0]

    if cfg.prompt1 == cfg.prompt2:
        deformation_embedding_index = -1
    else:
        deformation_index_found = False
        for i in range(77):
            if not torch.allclose(text_embeddings1[:,i,:], text_embeddings2[:,i,:]):
                deformation_embedding_index = i
                print(f"deformation embedding index: {i}")
                deformation_index_found = True
                break
        assert deformation_index_found

    max_length = text_input1.input_ids.shape[-1]
    uncond_input = tokenizer([""], padding="max_length",
         max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings1 = torch.cat([uncond_embeddings, text_embeddings1])
    text_embeddings2 = torch.cat([uncond_embeddings, text_embeddings2])

    ### weight loss
    num_train_timesteps = len(scheduler.betas)
    loss_weights = get_loss_weights(scheduler.betas, cfg)
    scheduler.set_timesteps(num_train_timesteps)

    ### instantiate model
    model = instantiate(cfg.model)
    model = model.to(device, dtype=dtype)
    model.train()

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_groups = model.parameter_groups(cfg.lr)
    print(f'Total number of trainable parameters: {total_parameters}')
    optimizer = torch.optim.Adam(model.parameter_groups(cfg.lr), lr=cfg.lr)

    train_loss_values = []
    ave_train_loss_values = []
    image_progress = []
    t_schedule = get_t_schedule(num_train_timesteps, cfg, loss_weights, device)
    pbar = tqdm(t_schedule)

    for step, t in enumerate(pbar):
        current_text_embeddings =\
                text_embeddings1 if step % 2 == 0 else text_embeddings2
        current_deformation_embedding =\
            current_text_embeddings[0,deformation_embedding_index]
        model_rgb, model_latents = get_outputs(model, vae,
               deformation_code=current_deformation_embedding, step=step)
        noise = torch.randn_like(model_latents)
        noisy_model_latents = scheduler.add_noise(model_latents, noise, t)
        optimizer.zero_grad()
        noise_pred = predict_noise(unet, noisy_model_latents, noise,
                    current_text_embeddings, t, \
                    guidance_scale=cfg.guidance_scale,
                    multisteps=cfg.multisteps, scheduler=scheduler,
                    half_inference=cfg.half_inference)

        loss = 0
        if cfg.loss.sds:
            grad = noise_pred - noise
            grad = torch.nan_to_num(grad)
            grad *= loss_weights[int(t)]
            target = (model_latents - grad).detach()
            sds_loss = 0.5 * F.mse_loss(model_latents,
                target, reduction="mean")
            loss = loss + cfg.loss.sds*sds_loss
        if cfg.loss.BGTplus:
            target_latents = scheduler.step(noise_pred, t,
                    noisy_model_latents).pred_original_sample.\
                            clone().detach().to(vae.dtype)
            target_rgb = vae.decode(target_latents / vae.config.scaling_factor
                    ).sample.clone().detach()
            BGTplus_loss = F.mse_loss(model_latents, target_latents,
                  reduction="mean") + 0.1*F.mse_loss(model_rgb,
                     target_rgb, reduction="mean")
            loss = loss + cfg.loss.BGTplus*BGTplus_loss

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        train_loss_values.append(loss.item())
        pbar.set_description(
                f'Loss: {loss.item():.6f}, sampled t : {t.item()}')
        optimizer.zero_grad()

        ######## Evaluation and log metric #########
        if cfg.log_steps and (step % cfg.log_steps == 0 or
                step == (cfg.num_steps-1)):
            target_latents = scheduler.step(noise_pred, t,
                    noisy_model_latents).pred_original_sample.to(dtype).clone().detach()
            noise_pred = noise_pred.clone().detach()
            with torch.no_grad():
                if cfg.half_inference:
                    target_latents = target_latents.half()
                target_rgb = vae.decode(target_latents / vae.config.scaling_factor
                        ).sample.to(torch.float32)
                log_img = torch.cat((model_rgb,target_rgb), dim=2)
            image_progress.append((log_img/2+0.5).clamp(0, 1))
            log_img_filename = f'{show_step(step, cfg.num_steps)}.png'
            save_image((log_img/2+0.5).clamp(0, 1),
                    os.path.join(output_dir, log_img_filename))

    images = sorted(Path(output_dir).glob(f"step*.png"))
    images = [imageio.imread(image) for image in images]
    writer = imageio.get_writer(os.path.join(output_dir, "progress.mp4"),
            fps=10, codec='mpeg4')
    for img in images:
        writer.append_data(img)
    writer.close()

#########################################################################################
if __name__ == "__main__":
    main()
