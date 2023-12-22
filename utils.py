import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging
import random
import os

SLURM_OUTPUT_DIR = f"J{os.getenv('SLURM_JOB_ID')}_{os.getenv('SLURM_JOB_NAME')}"

def seed_all(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

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

def show_step(step, total_steps):
    return f'step{str(step).zfill(len(str(total_steps)))}'

def get_t_schedule(num_train_timesteps, args, loss_weight=None, device="cpu"):
    ts = list(range(num_train_timesteps))
    assert (args.t_start >= 20) and (args.t_end <= 980)
    ts = ts[args.t_start:args.t_end]
    if args.t_schedule == 'random':
        chosen_ts = np.random.choice(ts, args.num_steps, replace=True)
    elif 'fixed' in args.t_schedule:
        fixed_t = int(args.t_schedule[5:])
        chosen_ts = [fixed_t for _ in range(args.num_steps)]
    elif 'descend' in args.t_schedule:
        descend_from = int(args.t_schedule[7:]) if len(args.t_schedule[7:]) > 0 else len(ts)
        chosen_ts = np.linspace(descend_from-1, 1, args.num_steps, endpoint=True)
        chosen_ts = chosen_ts.astype(int).tolist()
    else:
        raise ValueError(f"Unknown scheduling strategy: {args.t_schedule}")
    return torch.tensor(chosen_ts).unsqueeze(1).to(device)

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

def predict_noise(unet, noisy_latents, noise, text_embeddings,
      t, guidance_scale=7.5, multisteps=1, scheduler=None, half_inference = False):
    with torch.no_grad():
        if multisteps > 1:
            noise_pred = predict_noise0_diffuser_multistep(unet, noisy_latents, text_embeddings,
                       t, guidance_scale=guidance_scale, scheduler=scheduler,
                       steps=multisteps, eta=0., half_inference=half_inference)
        else:
            noise_pred = predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t,
                         guidance_scale=guidance_scale, scheduler=scheduler,
                         half_inference=half_inference)
    return noise_pred.to(unet.dtype)

def get_outputs(model, vae, **kwargs):
    model_out = model.generate(**kwargs)
    if model.distillation_space=="latent":
        model_latents = F.interpolate(model_out.unsqueeze(0), (64, 64), mode="bilinear", align_corners=False).to(vae.dtype)
        model_rgb = vae.decode(model_latents/vae.config.scaling_factor).sample
    elif model.distillation_space=="rgb":
        model_rgb = F.interpolate(model_out.unsqueeze(0), (512, 512), mode="bilinear", align_corners=False).to(vae.dtype)
        model_latents = vae.config.scaling_factor * vae.encode(model_rgb).latent_dist.sample()
    return model_rgb, model_latents
