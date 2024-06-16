from diffusers import StableDiffusionPipeline, DDIMScheduler
from omegaconf import OmegaConf
from utils import SLURM_OUTPUT_DIR
import hydra
import yaml
import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

@hydra.main(config_path="conf",
            config_name="config", version_base=None)
def main(cfg):
    if os.getenv("SLURM_JOB_ID"):
        output_dir = os.path.join("out", SLURM_OUTPUT_DIR)
    else:
        output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,"cfg.yaml"), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    device = "cuda"
    ddim = DDIMScheduler.from_pretrained(
           cfg.model_id, subfolder="scheduler",
           cache_dir=cfg.model_dir, local_files_only=cfg.local_files_only)
    pipe = StableDiffusionPipeline.from_pretrained(cfg.model_id,schedule=ddim,
           cache_dir=cfg.model_dir, local_files_only=cfg.local_files_only).to(device)
    del pipe.scheduler

    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    image = call_pipeline(
        pipe,
        scheduler=ddim,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        height=cfg.img_resolution, width=cfg.img_resolution,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator
    )[0]
    save_image(image, os.path.join(output_dir,"out.png"))

@torch.no_grad()
def call_pipeline(
    pipeline,
    scheduler,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    **kwargs,
):
    pipeline._cross_attention_kwargs = None

    device = pipeline._execution_device

    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    do_classifier_free_guidance=True,
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    latents = randn_tensor([1,4,64,64], generator=generator, device=device, dtype=prompt_embeds.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # predict the noise residual
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=pipeline.cross_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor,
                                return_dict=False, generator=generator)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

if __name__ == "__main__":
    main()
