import glob
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import argparse
import os
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as nnf
from torchvision import transforms
from torchvision.io import write_video
from torchvision.utils import save_image
import inspect
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler,PNDMScheduler, LMSDiscreteScheduler

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def backward_ddim(x_t, alpha_t: "alpha_t", alpha_tm1: "alpha_{t-1}", eps_xt):
    """ from noise to image"""
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )

def forward_ddim(x_t, alpha_t: "alpha_t", alpha_tp1: "alpha_{t+1}", eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)

class InversionStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker = None,
        feature_extractor: CLIPFeatureExtractor = None,
        guidance_args = None,
    ):
        super().__init__(vae,text_encoder,tokenizer, unet, scheduler, safety_checker, feature_extractor)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler
        )
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
        self.guidance_args = guidance_args
    
    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings

    @torch.enable_grad()
    def apply_guidance(self, tweedie, guidance_img, guidance_mask, num_steps=5):
        param = tweedie.clone().detach().requires_grad_(True)
        target = guidance_img.detach()
        optimizer = torch.optim.Adam([param], lr=self.guidance_args.guidance_lr)
        for i in range(num_steps):
                # Compute prediction and loss
                rgb_pred = self.decode_image(param.unsqueeze(0))
                loss = ((rgb_pred - target).abs()*guidance_mask).sum() / guidance_mask.sum()
                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return param.detach().clone(), loss.detach()
    
    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.no_grad()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: True = False,
        return_dict: bool = False,
        guide_tweedies: bool = False,
        guidance_frames: torch.FloatTensor = None,
        guidance_masks: torch.FloatTensor = None,
        guidance_frame_indices: List[int] = [0],
        **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False
        
        latents_dict = {}
        trajectory_dict = {}
        for frame_id in guidance_frame_indices:
            latents_dict[frame_id] = latents.clone()
            trajectory_dict[frame_id] = []

        del latents

        pbar = self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))
        for i, t in enumerate(pbar):
            for frame_id in guidance_frame_indices:
                if prompt_to_prompt:
                    if i < use_old_emb_i:
                        text_embeddings = old_text_embeddings
                    else:
                        text_embeddings = new_text_embeddings

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents_dict[frame_id]] * 2) if do_classifier_free_guidance else latents_dict[frame_id]
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
                )
                # call the callback, if provided
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                
                # ddim 
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep >= 0
                    else self.scheduler.final_alpha_cumprod
                )
                if reverse_process:
                    alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
                
                #estimate tweedie
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latents_dict[frame_id] - beta_prod_t ** (0.5) * noise_pred
                    ) / alpha_prod_t ** (0.5)

                trajectory_dict[frame_id].append(pred_original_sample.clone())

                if guide_tweedies:
                    with torch.enable_grad():
                        updated_tweedie, guidance_loss = self.apply_guidance(pred_original_sample.clone().squeeze(0),
                              guidance_frames[frame_id], guidance_masks[frame_id],
                              num_steps=self.guidance_args.num_guidance_steps)
                    pred_original_sample = updated_tweedie
                    pbar.set_description(f"Guidance Loss {guidance_loss.item():,.4f}")

                #compute the prev sample from the tweedie
                pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * noise_pred
                prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                latents_dict[frame_id] = prev_sample

        if return_dict:
            for k in trajectory_dict.keys():
                trajectory_dict[k] = torch.cat(trajectory_dict[k])
            return {'trajectory_dict': trajectory_dict, 'latents_dict': latents_dict}
        else:
            return latents_dict
    
    @torch.enable_grad()
    def decode_image(self, latents: torch.FloatTensor, **kwargs) -> List["PIL_IMAGE"]:
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image) -> List["PIL_IMAGE"]:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    @torch.inference_mode()
    def latents_to_imgs(self, latents):
        x = self.decode_image(latents)
        x = self.torch_to_numpy(x)
        x = self.numpy_to_pil(x)
        return x

    @torch.inference_mode()
    def save_latent_videoframes(self, latent_frames, filepath, fps=8):
        rgb_frames = 0.5 + 0.5*self.decode_image(latent_frames).cpu()
        rgb_frames = rgb_frames.clip(0,1)
        rgb_frames = rgb_frames.permute(0,2,3,1)*255
        write_video(filepath, rgb_frames, fps=fps)

def get_inversion_pipe(
    model_id="stabilityai/stable-diffusion-2-1-base",
    model_dir="hf-models", device="cuda", guidance_args=None):
        original_pipe = StableDiffusionPipeline.from_pretrained(model_id,
            cache_dir=model_dir).to(device)
        # run this when you modify the code
        pipe = InversionStableDiffusionPipeline(
            vae=original_pipe.vae,
            text_encoder=original_pipe.text_encoder,
            tokenizer=original_pipe.tokenizer,
            unet=original_pipe.unet,
            scheduler=original_pipe.scheduler,
            safety_checker=original_pipe.safety_checker,
            feature_extractor=original_pipe.feature_extractor,
            guidance_args=guidance_args
        )
        return pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--guidance_lr', type=float, default=1e-2)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--frame_id', type=int, default=[0], nargs='+')
    parser.add_argument('--num_guidance_steps', type=int, default=5)
    args = parser.parse_args()

    output_dir = f"out/inversion_{args.guidance_lr}_{args.num_inference_steps}_steps"
    os.makedirs(os.path.join(output_dir, "edit_trajectories"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "edited_frames"), exist_ok=True)

    pipe = get_inversion_pipe(guidance_args=args)
    context_img_path = Path("data/target_context_frame.png").expanduser()
    prompt = "A photo of a man in a room"
    text_embeddings = pipe.get_text_embedding(prompt)

    context_img = load_img(context_img_path).unsqueeze(0).to("cuda")
    context_latents = pipe.get_image_latents(context_img, rng_generator=torch.Generator(
        device=pipe.device).manual_seed(0))

    reversed_context_latents = pipe.forward_diffusion(
        latents=context_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
    )[0]

    guidance_frame_paths = sorted(glob.glob("data/warped_marigold_frames/*.png"))
    guidance_mask_paths = [p.replace("marigold_frames","marigold_mask_frames")\
            for p in guidance_frame_paths]
    guidance_frames = torch.stack([load_img(gp).to("cuda") for gp in guidance_frame_paths])
    guidance_masks = 0.5*torch.stack([load_img(gp).to("cuda") for gp in guidance_frame_paths]) + 0.5
    guidance_frame_indices = args.frame_id

    edit_recon_output_dict = pipe.backward_diffusion(
        latents=reversed_context_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
        return_dict=True,
        guide_tweedies=True,
        guidance_frames=guidance_frames,
        guidance_masks=guidance_masks,
        guidance_frame_indices = guidance_frame_indices
    )

    edit_recon_latents = edit_recon_output_dict['latents_dict']
    edit_recon_trajectory = edit_recon_output_dict['trajectory_dict']
    edited_frames = {}
    latent_video_out = []

    for frame_id in guidance_frame_indices:
        edited_frame = pipe.latents_to_imgs(edit_recon_latents[frame_id])[0]
        edited_frame.save(f"{output_dir}/edited_frames/{str(frame_id).zfill(2)}.png")
        latent_video_out.append(edit_recon_latents[frame_id])
        pipe.save_latent_videoframes(edit_recon_trajectory[frame_id],
            f"{output_dir}/edit_trajectories/{str(frame_id).zfill(2)}.mp4")

    if len(latent_video_out) > 1:
        pipe.save_latent_videoframes(torch.cat(latent_video_out), f"{output_dir}/out.mp4")
    else:
        out = pipe.latents_to_imgs(latent_video_out[0])[0]
        out.save(f"{output_dir}/out.png")
