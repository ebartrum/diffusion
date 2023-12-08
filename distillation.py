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
            get_optimizer,
            )
import shutil
import logging
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

    ### set up logger
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.basicConfig(filename=f'{output_dir}/experiment.log', filemode='w',
                        format='%(asctime)s %(levelname)s --> %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(f'[INFO] Cmdline: '+' '.join(sys.argv))

    ### log basic info
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

    if cfg.generation_mode == 'vsd':
        if cfg.phi_model == 'lora':
            if cfg.lora_vprediction:
                assert cfg.model_path == 'stabilityai/stable-diffusion-2-1-base'
                vae_phi = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="vae", torch_dtype=dtype).to(device)
                unet_phi = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="unet", torch_dtype=dtype).to(device)
                vae_phi.requires_grad_(False)
                unet_phi, unet_lora_layers = extract_lora_diffusers(unet_phi, device)
            else:
                vae_phi = vae
                ### unet_phi is the same instance as unet that has been modified in-place
                unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)
            phi_params = list(unet_lora_layers.parameters())
            if cfg.load_phi_model_path:
                unet_phi.load_attn_procs(cfg.load_phi_model_path)
                unet_phi = unet_phi.to(device)
        elif cfg.phi_model == 'unet_simple':
            # initialize simple unet, same input/output as (pre-trained) unet
            ### IMPORTANT: need the proper (wide) channel numbers
            channels = 4 if cfg.rgb_as_latents else 3
            unet_phi = UNet2DConditionModel(
                                        sample_size=64,
                                        in_channels=channels,
                                        out_channels=channels,
                                        layers_per_block=1,
                                        block_out_channels=(64,128,256),
                                        down_block_types=(
                                            "CrossAttnDownBlock2D",
                                            "CrossAttnDownBlock2D",
                                            "DownBlock2D",
                                        ),
                                        up_block_types=(
                                            "UpBlock2D",
                                            "CrossAttnUpBlock2D",
                                            "CrossAttnUpBlock2D",
                                        ),
                                        cross_attention_dim=unet.config.cross_attention_dim,
                                        ).to(dtype)
            if cfg.load_phi_model_path:
                unet_phi = unet_phi.from_pretrained(cfg.load_phi_model_path)
            unet_phi = unet_phi.to(device)
            phi_params = list(unet_phi.parameters())
            vae_phi = vae
    elif cfg.generation_mode == 'sds':
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

    ### scheduler set timesteps
    if cfg.generation_mode == 't2i':
        scheduler.set_timesteps(cfg.num_steps)
    else:
        scheduler.set_timesteps(num_train_timesteps)

    ### initialize particles
    if cfg.use_mlp_particle:
        # use siren network
        from model_utils import Siren
        cfg.lr = 1e-4
        print(f'for mlp_particle, set lr to {cfg.lr}')
        out_features = 4 if cfg.rgb_as_latents else 3
        particles = nn.ModuleList([Siren(2, hidden_features=256, hidden_layers=3, out_features=out_features, device=device) for _ in range(cfg.batch_size)])
    else:
        if cfg.init_img_path:
            # load image
            init_image = io.read_image(cfg.init_img_path).unsqueeze(0) / 255
            init_image = init_image * 2 - 1   #[-1,1]
            if cfg.rgb_as_latents:
                particles = vae.config.scaling_factor * vae.encode(init_image.to(device)).latent_dist.sample()
            else:
                particles = init_image.to(device)
        else:
            if cfg.rgb_as_latents:
                particles = torch.randn((cfg.batch_size, unet.config.in_channels, cfg.height // 8, cfg.width // 8))
            else:
                # gaussian in rgb space --> strange artifacts
                particles = torch.randn((cfg.batch_size, 3, cfg.height, cfg.width))
                cfg.lr = cfg.lr * 1   # need larger lr for rgb particles
                # ## gaussian in latent space --> not better
                # particles = torch.randn((cfg.batch_size, unet.in_channels, cfg.height // 8, cfg.width // 8)).to(device, dtype=dtype)
                # particles = vae.decode(particles).sample
    particles = particles.to(device, dtype=dtype)
    if cfg.nerf_init and cfg.rgb_as_latents and not cfg.use_mlp_particle:
        # current only support sds and experimental for only rgb_as_latents==True
        assert cfg.generation_mode == 'sds'
        with torch.no_grad():
            noise_pred = predict_noise0_diffuser(unet, particles, text_embeddings_vsd, t=999, guidance_scale=7.5, scheduler=scheduler)
        particles = scheduler.step(noise_pred, 999, particles).pred_original_sample
    #######################################################################################
    ### configure optimizer and loss function
    if cfg.use_mlp_particle:
        # For a list of models, we want to optimize their parameters
        particles_to_optimize = [param for mlp in particles for param in mlp.parameters() if param.requires_grad]
    else:
        # For a tensor, we can optimize the tensor directly
        particles.requires_grad = True
        particles_to_optimize = [particles]

    total_parameters = sum(p.numel() for p in particles_to_optimize if p.requires_grad)
    print(f'Total number of trainable parameters in particles: {total_parameters}; number of particles: {cfg.batch_size}')
    ### Initialize optimizer & scheduler
    if cfg.generation_mode == 'vsd':
        if cfg.phi_model in ['lora', 'unet_simple']:
            phi_optimizer = torch.optim.AdamW([{"params": phi_params, "lr": cfg.phi_lr}], lr=cfg.phi_lr)
            print(f'number of trainable parameters of phi model in optimizer: {sum(p.numel() for p in phi_params if p.requires_grad)}')
    optimizer = get_optimizer(particles_to_optimize, cfg)
    if cfg.use_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, \
            start_factor=cfg.lr_scheduler_start_factor, total_iters=cfg.lr_scheduler_iters)

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
    ### regular sd text to image generation
    if cfg.generation_mode == 't2i':
        if cfg.phi_model == 'lora' and cfg.load_phi_model_path:
            ### unet_phi is the same instance as unet that has been modified in-place
            unet_phi, unet_lora_layers = extract_lora_diffusers(unet, device)
            phi_params = list(unet_lora_layers.parameters())
            unet_phi.load_attn_procs(cfg.load_phi_model_path)
            unet = unet_phi.to(device)
        step = 0
        # get latent of all particles
        assert cfg.use_mlp_particle == False
        latents = get_latents(particles, vae, cfg.rgb_as_latents)
        if cfg.half_inference:
            latents = latents.half()
            text_embeddings_vsd = text_embeddings_vsd.half()
        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            latent_noisy = latents
            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_vsd).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            ######## Evaluation and log metric #########
            if cfg.log_steps and (step % cfg.log_steps == 0 or step == (cfg.num_steps-1)):
                # save current img_tensor
                # scale and decode the image latents with vae
                tmp_latents = 1 / vae.config.scaling_factor * latents.clone().detach()
                if cfg.save_x0:
                    # compute the predicted clean sample x_0
                    pred_latents = scheduler.step(noise_pred, t, latent_noisy).pred_original_sample.to(dtype).clone().detach()
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
            step += 1
    ### sds text to image generation
    elif cfg.generation_mode in ['sds', 'vsd']:
        cross_attention_kwargs = {'scale': cfg.lora_scale} if (cfg.generation_mode == 'vsd' and cfg.phi_model == 'lora') else {}
        for step, chosen_t in enumerate(pbar):
            # get latent of all particles
            latents = get_latents(particles, vae, cfg.rgb_as_latents, use_mlp_particle=cfg.use_mlp_particle)
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
            if cfg.use_scheduler:
                lr_scheduler.step(loss)

            torch.cuda.empty_cache()
            ######## Do the gradient for unet_phi!!! #########
            if cfg.generation_mode == 'vsd':
                ## update the unet (phi) model
                for _ in range(cfg.phi_update_step):
                    phi_optimizer.zero_grad()
                    if cfg.use_t_phi:
                        # different t for phi finetuning
                        # t_phi = np.random.choice(chosen_ts, 1, replace=True)[0]
                        t_phi = np.random.choice(list(range(num_train_timesteps)), 1, replace=True)[0]
                        t_phi = torch.tensor([t_phi]).to(device)
                    else:
                        t_phi = t
                    # random sample particle_num_phi particles from latents
                    indices = torch.randperm(latents.size(0))
                    latents_phi = latents[indices[:cfg.particle_num_phi]]
                    noise_phi = torch.randn_like(latents_phi)
                    noisy_latents_phi = scheduler.add_noise(latents_phi, noise_phi, t_phi)
                    loss_phi = phi_vsd_grad_diffuser(unet_phi, noisy_latents_phi.detach(), noise_phi, text_embeddings_phi, t_phi, \
                                                     cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, \
                                                        lora_v=cfg.lora_vprediction, half_inference=cfg.half_inference)

                    loss_phi.backward()
                    phi_optimizer.step()

            ### Store loss and step
            train_loss_values.append(loss.item())
            ### update pbar
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
                    if cfg.generation_mode == 'vsd':
                        pred_latents_phi = scheduler.step(noise_pred_phi, t, noisy_latents).pred_original_sample.to(dtype).clone().detach()
                with torch.no_grad():
                    if cfg.half_inference:
                        tmp_latents = tmp_latents.half()
                    image_ = vae.decode(tmp_latents).sample.to(torch.float32)
                    if cfg.save_x0:
                        if cfg.half_inference:
                            pred_latents = pred_latents.half()
                        image_x0 = vae.decode(pred_latents / vae.config.scaling_factor).sample.to(torch.float32)
                        if cfg.generation_mode == 'vsd':
                            if cfg.half_inference:
                                pred_latents_phi = pred_latents_phi.half()
                            image_x0_phi = vae_phi.decode(pred_latents_phi / vae.config.scaling_factor).sample.to(torch.float32)
                            image = torch.cat((image_,image_x0,image_x0_phi), dim=2)
                        else:
                            image = torch.cat((image_,image_x0), dim=2)
                    else:
                        image = image_
                if cfg.log_progress:
                    image_progress.append((image/2+0.5).clamp(0, 1))
                save_image((image/2+0.5).clamp(0, 1), f'{output_dir}/{image_name}_image_step{step}_t{t.item()}.png')

    if cfg.log_gif:
        # make gif
        images = sorted(Path(output_dir).glob(f"*{image_name}*.png"))
        images = [imageio.imread(image) for image in images]
        imageio.mimsave(f'{output_dir}/{image_name}.gif', images, duration=0.3)
    if cfg.log_progress and cfg.batch_size == 1:
        concatenated_images = torch.cat(image_progress, dim=0)
        save_image(concatenated_images, f'{output_dir}/{image_name}_prgressive.png')
    # save final image
    if cfg.generation_mode == 't2i':
        image = image_
    else:
        image = get_images(particles, vae, cfg.rgb_as_latents, use_mlp_particle=cfg.use_mlp_particle)
    save_image((image/2+0.5).clamp(0, 1), f'{output_dir}/final_image_{image_name}.png')

    if cfg.generation_mode in ['vsd'] and cfg.save_phi_model:
        if cfg.phi_model in ['lora']:
            unet_phi.save_attn_procs(save_directory=f'{output_dir}')
        elif cfg.phi_model in ['unet_simple']:
            unet_phi.save_pretrained(save_directory=f'{output_dir}')

#########################################################################################
if __name__ == "__main__":
    main()
