from diffusers import StableDiffusionPipeline, DDIMScheduler
from omegaconf import OmegaConf
import hydra
import yaml
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor, center_crop
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image

@hydra.main(config_path="conf",
            config_name="config", version_base=None)
def main(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir,"cfg.yaml"), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda:0")
    ddim = DDIMScheduler.from_pretrained(
           cfg.model_id, subfolder="scheduler",
           cache_dir=cfg.model_dir, local_files_only=cfg.local_files_only)
    pipe = StableDiffusionPipeline.from_pretrained(cfg.model_id,schedule=ddim,
           cache_dir=cfg.model_dir, local_files_only=cfg.local_files_only).to(device)
    del pipe.scheduler

    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    if cfg.inversion:
        input_image_path = "~/Documents/repos/diffusion/data/frontal_face.jpg"
        img = Image.open("./data/frontal_face.jpg").convert("RGB")
        img = to_tensor(img).to(device)
        img = center_crop(img,min(img.shape[1],img.shape[2]))
        img = F.interpolate(img.unsqueeze(0),512).squeeze(0)
        with torch.no_grad():
            latent = pipe.vae.encode(img.unsqueeze(0)*2 - 1)
        z0 = pipe.vae.config.scaling_factor * latent.latent_dist.sample()
        inverted_latents, inversion_logging_trajectory = invert(
              z0,
              pipe,
              device,
              scheduler=ddim,
              prompt=cfg.prompt,
              negative_prompt=cfg.negative_prompt,
              guidance_scale=cfg.guidance_scale,
              guidance_mode=cfg.guidance_mode,
              num_inference_steps=cfg.num_inference_steps,
              generator=generator,
              )

        final_inverted_latent = inverted_latents[-1]
        input_latent = final_inverted_latent
    else:
        input_latent = torch.randn((1,4,64,64), device=device)
    
    clean_latents, trajectory = denoise_latents(
        input_latent,
        pipe,
        device,
        scheduler=ddim,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        guidance_scale=cfg.guidance_scale,
        guidance_mode=cfg.guidance_mode,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator
    )

    image = latents2img(clean_latents, pipe, generator)
    save_image(image, os.path.join(cfg.output_dir,cfg.output_file))

    with torch.no_grad():
        img_trajectory = [latents2img(l, pipe, generator) for l in trajectory.values()]
        img_trajectory = [F.interpolate(img, 128) for img in img_trajectory]
        img_trajectory = torch.cat(img_trajectory, -1)
        img_trajectory_output_file = cfg.output_file.replace('.','_trajectory.')
        save_image(img_trajectory, os.path.join(cfg.output_dir,
                img_trajectory_output_file))

        if cfg.inversion:
            inversion_img_trajectory = [latents2img(l, pipe, generator) for l
                in inversion_logging_trajectory.values()]
            inversion_img_trajectory = [F.interpolate(img, 128) for img in inversion_img_trajectory]
            inversion_img_trajectory = torch.cat(inversion_img_trajectory, -1)
            inversion_img_trajectory_output_file = cfg.output_file.replace('.','_invert_trajectory.')
            save_image(inversion_img_trajectory, os.path.join(cfg.output_dir,
                    inversion_img_trajectory_output_file))

def latents2img(latents, pipe, generator):
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor,
        return_dict=False, generator=generator)[0]
    image = (image/2 + 0.5).clamp(0, 1)
    return image

def denoising_step(
        scheduler,
        pred_epsilon: torch.Tensor,
        pred_epsilon_uncond: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        guidance_mode: str = "cfg",
    ):
        # get previous step value (=t-1)
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

        # compute alphas, betas
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)

        # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if guidance_mode == "cfg":
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
        elif guidance_mode == "cfgpp":
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon_uncond
        else:
            raise ValueError(f"Unknown guidance_mode {guidance_mode}!")

        # compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return {'prev_sample': prev_sample,
                'pred_original_sample': pred_original_sample}

def inversion_step(
        scheduler,
        pred_epsilon: torch.Tensor,
        pred_epsilon_uncond: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        guidance_mode: str = "cfg",
    ):
        print(f"inversion_step: {timestep}")
        # get previous step value (=t-1)
        next_timestep = timestep + scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        if next_timestep > 999:
            next_timestep = 999

        # compute alphas, betas
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]

        beta_prod_t = 1 - alpha_prod_t

        if guidance_mode == "cfg":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)
        elif guidance_mode == "cfgpp":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon_uncond) / alpha_prod_t ** (0.5)
        else:
            raise ValueError(f"Unknown guidance_mode {guidance_mode}!")

        # compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * pred_epsilon

        next_sample = alpha_prod_t_next ** (0.5) * pred_original_sample + pred_sample_direction

        return {'next_sample': next_sample,
                'pred_original_sample': pred_original_sample}
@torch.no_grad()
def invert(
          z0, pipeline, device,
          scheduler,
          prompt: Union[str, List[str]] = None,
          negative_prompt: Optional[Union[str, List[str]]] = None,
          num_inference_steps: int = 50,
          guidance_scale: float = 7.5,
          guidance_mode: str = "cfg",
          do_classifier_free_guidance=True,
          generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
          save_trajectory_len=5,
          ):

    logging_trajectory = OrderedDict() 
    inversion_trajectory = [z0]
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    latents = z0.clone()

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, timesteps=None, sigmas=None
    )
    reverse_timesteps = reversed(timesteps)
    trajectory_log_indices = \
        torch.linspace(0,num_inference_steps,
           save_trajectory_len+1).round().int()[:-1]

    for i, t in enumerate(reverse_timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # predict the noise residual
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        inversion_output = inversion_step(scheduler, noise_pred, noise_pred_uncond,
             t, latents, guidance_mode=guidance_mode)
        latents = inversion_output['next_sample']
        inversion_trajectory.append(latents.clone())
        if i in trajectory_log_indices:
            logging_trajectory[t.item()] = inversion_output['pred_original_sample']

    return inversion_trajectory, logging_trajectory

@torch.no_grad()
def denoise_latents(
    latents,
    pipeline,
    device,
    scheduler,
    prompt: Union[str, List[str]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    guidance_mode: str = "cfg",
    negative_prompt: Optional[Union[str, List[str]]] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    do_classifier_free_guidance=True,
    save_trajectory_len=5,
):

    trajectory = OrderedDict() 
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )

    # Concatenate unconditional and conditional embeddings
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, timesteps=None, sigmas=None
    )
    trajectory_log_indices = \
        torch.linspace(0,num_inference_steps,
           save_trajectory_len+1).round().int()[:-1]

    # Denoise the latents
    pbar = tqdm(enumerate(timesteps), total=len(timesteps))
    for i, t in pbar:
        pbar.set_description(f"t: {t}")
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # predict the noise residual
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        ddim_output = denoising_step(scheduler, noise_pred, noise_pred_uncond,
             t, latents, guidance_mode=guidance_mode)
        latents = ddim_output['prev_sample']

        if i in trajectory_log_indices:
            trajectory[t.item()] = ddim_output['pred_original_sample']

    return latents, trajectory

if __name__ == "__main__":
    main()
