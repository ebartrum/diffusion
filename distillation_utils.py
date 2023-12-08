import torch
from torch import nn
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch.nn.functional as F
import logging
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    SlicedAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers

def setup_logger(output_dir):
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.basicConfig(filename=f'{output_dir}/experiment.log', filemode='w',
                        format='%(asctime)s %(levelname)s --> %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger

def get_t_schedule(num_train_timesteps, args, loss_weight=None):
    # Create a list of time steps from 0 to num_train_timesteps
    ts = list(range(num_train_timesteps))
    # set ts to U[0.02,0.98] as least
    assert (args.t_start >= 20) and (args.t_end <= 980)
    ts = ts[args.t_start:args.t_end]

    # If the scheduling strategy is 'random', choose args.num_steps random time steps without replacement
    if args.t_schedule == 'random':
        chosen_ts = np.random.choice(ts, args.num_steps, replace=True)

    # If the scheduling strategy is 'random_down', first exclude the first 30 and last 10 time steps
    # then choose a random time step from an interval that shrinks as step increases
    elif 'random_down' in args.t_schedule:
        interval_ratio = int(args.t_schedule[11:]) if len(args.t_schedule[11:]) > 0 else 5
        interval_ratio *= 0.1
        chosen_ts = [np.random.choice(
                        ts[max(0,int((args.num_steps-step-interval_ratio*args.num_steps)/args.num_steps*len(ts))):\
                           min(len(ts),int((args.num_steps-step+interval_ratio*args.num_steps)/args.num_steps*len(ts)))],
                     1, replace=True).astype(int)[0] \
                     for step in range(args.num_steps)]

    # If the scheduling strategy is 'fixed', parse the fixed time step from the string and repeat it args.num_steps times
    elif 'fixed' in args.t_schedule:
        fixed_t = int(args.t_schedule[5:])
        chosen_ts = [fixed_t for _ in range(args.num_steps)]

    # If the scheduling strategy is 'descend', parse the start time step from the string (or default to 1000)
    # then create a list of descending time steps from the start to 0, with length args.num_steps
    elif 'descend' in args.t_schedule:
        if 'quad' in args.t_schedule:   # no significant improvement
            descend_from = int(args.t_schedule[12:]) if len(args.t_schedule[7:]) > 0 else len(ts)
            chosen_ts = np.square(np.linspace(descend_from**0.5, 1, args.num_steps))
            chosen_ts = chosen_ts.astype(int).tolist()
        else:
            descend_from = int(args.t_schedule[7:]) if len(args.t_schedule[7:]) > 0 else len(ts)
            chosen_ts = np.linspace(descend_from-1, 1, args.num_steps, endpoint=True)
            chosen_ts = chosen_ts.astype(int).tolist()

    # If the scheduling strategy is 't_stages', the total number of time steps are divided into several stages.
    # In each stage, a decreasing portion of the total time steps is considered for selection.
    # For each stage, time steps are randomly selected with replacement from the respective portion.
    # The final list of chosen time steps is a concatenation of the time steps selected in all stages.
    # Note: The total number of time steps should be evenly divisible by the number of stages.
    elif 't_stages' in args.t_schedule:
        # Parse the number of stages from the scheduling strategy string
        num_stages = int(args.t_schedule[8:]) if len(args.t_schedule[8:]) > 0 else 2
        chosen_ts = []
        for i in range(num_stages):
            # Define the portion of ts to be considered in this stage
            portion = ts[:int((num_stages-i)*len(ts)//num_stages)]
            selected_ts = np.random.choice(portion, args.num_steps//num_stages, replace=True).tolist()
            chosen_ts += selected_ts

    elif 'dreamtime' in args.t_schedule:
        # time schedule in dreamtime https://arxiv.org/abs//2306.12422
        assert 'dreamtime' in args.loss_weight_type
        loss_weight_sum = np.sum(loss_weight)
        p = [wt / loss_weight_sum for wt in loss_weight]
        N = args.num_steps
        def t_i(t, i, p):
            t = int(max(0, min(len(p)-1, t)))
            return abs(sum(p[t:]) - i/N)
        chosen_ts = []
        for i in range(N):
            # Initial guess for t
            t0 = 999
            selected_t = minimize(t_i, t0, args=(i, p), method='Nelder-Mead')
            selected_t = max(0, int(selected_t.x))
            chosen_ts.append(selected_t)
    else:
        raise ValueError(f"Unknown scheduling strategy: {args.t_schedule}")

    # Return the list of chosen time steps
    return chosen_ts

def get_loss_weights(betas, args):
    num_train_timesteps = len(betas)
    betas = torch.tensor(betas) if not torch.is_tensor(betas) else betas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image
    sigma_ks = []
    SNRs = []
    rhos = []
    m1 = 800
    m2 = 500
    s1 = 300
    s2 = 100
    for i in range(num_train_timesteps):
        sigma_ks.append(reduced_alpha_cumprod[i])
        SNRs.append(1 / reduced_alpha_cumprod[i])
        if args.loss_weight_type == 'rhos':
            rhos.append(1. * (args.sigma_y**2)/(sigma_ks[i]**2))
    def loss_weight(t):
        if args.loss_weight_type == None or args.loss_weight_type == 'none':
            return 1
        elif 'SNR' in args.loss_weight_type:
            ## ref: https://arxiv.org/abs/2305.04391
            if args.loss_weight_type == 'SNR':
                return 1 / SNRs[t]
            elif args.loss_weight_type == 'SNR_sqrt':
                return torch.sqrt(1 / SNRs[t])
            elif args.loss_weight_type == 'SNR_square':
                return (1 / SNRs[t])**2
            elif args.loss_weight_type == 'SNR_log1p':
                return torch.log(1 + 1 / SNRs[t])
        elif args.loss_weight_type == 'rhos':
            return 1 / rhos[t]
        elif 'alpha' in args.loss_weight_type:
            if args.loss_weight_type == 'sqrt_alphas_cumprod':
                return sqrt_alphas_cumprod[t]
            elif args.loss_weight_type == '1m_alphas_cumprod':
                return sqrt_1m_alphas_cumprod[t]**2
            elif args.loss_weight_type == 'alphas_cumprod':
                return alphas_cumprod[t]
            elif args.loss_weight_type == 'sqrt_alphas_1m_alphas_cumprod':
                return sqrt_alphas_cumprod[t] * sqrt_1m_alphas_cumprod[t]
        elif 'dreamtime' in args.loss_weight_type:
            if t > m1:
                return np.exp(-(t - m1)**2 / (2 * s1**2))
            elif t >= m2:
                return 1
            else:
                return np.exp(-(t - m2)**2 / (2 * s2**2))
        else:
            raise NotImplementedError
    weights = []
    for i in range(num_train_timesteps):
        weights.append(loss_weight(i))
    return weights

def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, half_inference=False):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # Convert inputs to half precision
    if half_inference:
        noisy_latents = noisy_latents.clone().half()
        text_embeddings = text_embeddings.clone().half()
        latent_model_input = latent_model_input.clone().half()
    if guidance_scale == 1.:
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cross_attention_kwargs).sample
    else:
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise_pred = noise_pred.float()
    return noise_pred

def predict_noise0_diffuser_multistep(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, steps=1, eta=0, half_inference=False):
    latents = noisy_latents
    # get sub-sequence with length step_size
    t_start = t.item()
    # Ensure that t and steps are within the valid range
    if not (0 < t_start <= 1000):
        raise ValueError(f"t must be between 0 and 1000, get {t_start}")
    if t_start > steps:
        # Calculate the step size
        step_size = t_start // steps
        # Generate a list of indices
        indices = [int((steps - i) * step_size) for i in range(steps)]
        if indices[0] != t_start:
            indices[0] = t_start    # replace start point
    else:
        indices = [int((t_start - i)) for i in range(t_start)]
    if indices[-1] != 0:
        indices.append(0)
    # run multistep ddim sampling
    for i in range(len(indices)):
        t = torch.tensor([indices[i]] * t.shape[0], device=t.device)
        noise_pred = predict_noise0_diffuser(unet, latents, text_embeddings, t, guidance_scale=guidance_scale, \
                                             cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, half_inference=half_inference)
        pred_latents = scheduler.step(noise_pred, t, latents).pred_original_sample
        if indices[i+1] == 0:
            ### use pred_latents and latents calculate equivalent noise_pred
            alpha_bar_t_start = scheduler.alphas_cumprod[indices[0]].clone().detach()
            return (noisy_latents - torch.sqrt(alpha_bar_t_start)*pred_latents) / (torch.sqrt(1 - alpha_bar_t_start))
        alpha_bar = scheduler.alphas_cumprod[indices[i]].clone().detach()
        alpha_bar_prev = scheduler.alphas_cumprod[indices[i+1]].clone().detach()
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(latents)
        mean_pred = (
            pred_latents * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(latents.shape) - 1)))
        )  # no noise when t == 0
        latents = mean_pred + nonzero_mask * sigma * noise

def sds_vsd_grad_diffuser(unet, noisy_latents, noise, text_embeddings, t, guidance_scale=7.5, \
                            multisteps=1, scheduler=None,
                                half_inference = False):
    unet_cross_attention_kwargs = {}
    with torch.no_grad():
        if multisteps > 1:
            noise_pred = predict_noise0_diffuser_multistep(unet, noisy_latents, text_embeddings, t, guidance_scale=guidance_scale, cross_attention_kwargs=unet_cross_attention_kwargs, scheduler=scheduler, steps=multisteps, eta=0., half_inference=half_inference)
        else:
            noise_pred = predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=guidance_scale, cross_attention_kwargs=unet_cross_attention_kwargs, scheduler=scheduler, half_inference=half_inference)

    grad = noise_pred - noise
    grad = torch.nan_to_num(grad)

    return grad, noise_pred.detach().clone()

def get_latents(particles, vae, rgb_as_latents=False):
    if rgb_as_latents:
        latents = F.interpolate(particles, (64, 64), mode="bilinear", align_corners=False)
    else:
        rgb_BCHW_512 = F.interpolate(particles, (512, 512), mode="bilinear", align_corners=False)
        latents = vae.config.scaling_factor * vae.encode(rgb_BCHW_512).latent_dist.sample()
    return latents

@torch.no_grad()
def batch_decode_vae(latents, vae):
    latents = 1 / vae.config.scaling_factor * latents.clone().detach()
    bs = 8  # avoid OOM for too many particles
    images = []
    for i in range(int(np.ceil(latents.shape[0] / bs))):
        batch_i = latents[i*bs:(i+1)*bs]
        image_i = vae.decode(batch_i).sample.to(torch.float32)
        images.append(image_i)
    image = torch.cat(images, dim=0)
    return image

@torch.no_grad()
def get_images(particles, vae, rgb_as_latents=False):
    if rgb_as_latents:
        latents = F.interpolate(particles, (64, 64), mode="bilinear", align_corners=False)
        images = batch_decode_vae(latents, vae)
    else:
        images = F.interpolate(particles, (512, 512), mode="bilinear", align_corners=False)
    return images
