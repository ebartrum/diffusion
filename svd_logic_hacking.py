import torch 
import PIL.Image
from typing import Callable, Dict, List, Optional, Union, Tuple
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion \
    import StableVideoDiffusionPipelineOutput, _append_dims, tensor2vid
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput

def new_step(
    scheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    flow: Optional[torch.FloatTensor] = None,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
) -> Union[EulerDiscreteSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        s_churn (`float`):
        s_tmin  (`float`):
        s_tmax  (`float`):
        s_noise (`float`, defaults to 1.0):
            Scaling factor for noise added to the sample.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`):
            Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
            tuple.

    Returns:
        [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
            returned, otherwise a tuple is returned where the first element is the sample tensor.
    """

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    if not scheduler.is_scale_input_called:
        logger.warning(
            "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
            "See `StableDiffusionPipeline` for a usage example."
        )

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    sigma = scheduler.sigmas[scheduler.step_index]

    gamma = min(s_churn / (len(scheduler.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

    noise = randn_tensor(
        model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
    )

    eps = noise * s_noise
    sigma_hat = sigma * (gamma + 1)

    if gamma > 0:
        sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    # NOTE: "original_sample" should not be an expected prediction_type but is left in for
    # backwards compatibility
    if scheduler.config.prediction_type == "original_sample" or scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "epsilon":
        pred_original_sample = sample - sigma_hat * model_output
    elif scheduler.config.prediction_type == "v_prediction":
        # denoised = model_output * c_out + input * c_skip
        pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
        )

    #New logic here; apply the flow. For now, we can use a lambda of 0.5 for the interpolation.
    import ipdb;ipdb.set_trace()

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma_hat

    dt = scheduler.sigmas[scheduler.step_index + 1] - sigma_hat

    prev_sample = sample + derivative * dt

    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    scheduler._step_index += 1

    if not return_dict:
        return (prev_sample,)

    return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

@torch.no_grad()
def new_call(
    pipe,
    image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
    height: int = 576,
    width: int = 1024,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 25,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: Optional[int] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    return_dict: bool = True,
    flow: Optional[torch.tensor] = None,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
            Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0, 1]`.
        height (`int`, *optional*, defaults to `pipe.unet.config.sample_size * pipe.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `pipe.unet.config.sample_size * pipe.vae_scale_factor`):
            The width in pixels of the generated image.
        num_frames (`int`, *optional*):
            The number of video frames to generate. Defaults to `pipe.unet.config.num_frames`
            (14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
        num_inference_steps (`int`, *optional*, defaults to 25):
            The number of denoising steps. More denoising steps usually lead to a higher quality video at the
            expense of slower inference. This parameter is modulated by `strength`.
        min_guidance_scale (`float`, *optional*, defaults to 1.0):
            The minimum guidance scale. Used for the classifier free guidance with first frame.
        max_guidance_scale (`float`, *optional*, defaults to 3.0):
            The maximum guidance scale. Used for the classifier free guidance with last frame.
        fps (`int`, *optional*, defaults to 7):
            Frames per second. The rate at which the generated images shall be exported to a video after generation.
            Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
        motion_bucket_id (`int`, *optional*, defaults to 127):
            Used for conditioning the amount of motion for the generation. The higher the number the more motion
            will be in the video.
        noise_aug_strength (`float`, *optional*, defaults to 0.02):
            The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
        decode_chunk_size (`int`, *optional*):
            The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the expense of more memory usage. By default, the decoder decodes all frames at once for maximal
            quality. For lower memory usage, reduce `decode_chunk_size`.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of videos to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `pil`, `np` or `pt`.
        callback_on_step_end (`Callable`, *optional*):
            A function that is called at the end of each denoising step during inference. The function is called
            with the following arguments:
                `callback_on_step_end(pipe: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
            `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
            otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.FloatTensor`) is returned.
    """
    # 0. Default height and width to unet
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    num_frames = num_frames if num_frames is not None else pipe.unet.config.num_frames
    decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(image, height, width)

    # 2. Define call parameters
    if isinstance(image, PIL.Image.Image):
        batch_size = 1
    elif isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]
    device = pipe._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    pipe._guidance_scale = max_guidance_scale

    # 3. Encode input image
    image_embeddings = pipe._encode_image(image, device, num_videos_per_prompt, pipe.do_classifier_free_guidance)

    # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
    # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
    fps = fps - 1

    # 4. Encode input image using VAE
    image = pipe.image_processor.preprocess(image, height=height, width=width).to(device)
    noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
    image = image + noise_aug_strength * noise

    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float32)

    image_latents = pipe._encode_vae_image(
        image,
        device=device,
        num_videos_per_prompt=num_videos_per_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
    )
    image_latents = image_latents.to(image_embeddings.dtype)

    # cast back to fp16 if needed
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    # Repeat the image latents for each frame so we can concatenate them with the noise
    # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
    image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

    # 5. Get Added Time IDs
    added_time_ids = pipe._get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        image_embeddings.dtype,
        batch_size,
        num_videos_per_prompt,
        pipe.do_classifier_free_guidance,
    )
    added_time_ids = added_time_ids.to(device)

    # 6. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 7. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    latents = pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_frames,
        num_channels_latents,
        height,
        width,
        image_embeddings.dtype,
        device,
        generator,
        latents,
    )

    # 8. Prepare guidance scale
    guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
    guidance_scale = guidance_scale.to(device, latents.dtype)
    guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
    guidance_scale = _append_dims(guidance_scale, latents.ndim)

    pipe._guidance_scale = guidance_scale

    # 9. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate image_latents over channels dimension
            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

            # predict the noise residual
            unet_out = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=True,
            )
            noise_pred = unet_out['sample']

            # perform guidance
            if pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            step_output = new_step(pipe.scheduler, noise_pred, t, latents, flow=flow)
            latents, tweedie_estimate = step_output.prev_sample, step_output.pred_original_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    if not output_type == "latent":
        # cast back to fp16 if needed
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        frames = pipe.decode_latents(latents, num_frames, decode_chunk_size)
        frames = tensor2vid(frames, pipe.image_processor, output_type=output_type)
    else:
        frames = latents

    pipe.maybe_free_model_hooks()

    if not return_dict:
        return frames

    return StableVideoDiffusionPipelineOutput(frames=frames)
