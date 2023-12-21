import os
import torch
from torchvision.utils import save_image
from utils import get_outputs, predict_noise

class VisualisationLogger():
    def __init__(self, cfg, model, unet, vae, scheduler, text_embeddings,
                 output_dir, log_steps=50, total_steps=1000):
        self.name = "img_log"
        self.cfg = cfg
        self.model = model
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.text_embeddings = text_embeddings
        self.log_steps = log_steps
        self.total_steps = total_steps
        self.output_dir = output_dir

    def is_log_step(self, step):
        out = self.log_steps and (step % self.log_steps == 0 or
                step == (self.total_steps-1))
        return out

    def log_img(self, step, t):
        with torch.no_grad():
            log_img = self.generate_vis_img(step, t)
        log_img_filename = f"{self.name}_{step}.png"
        save_image((log_img/2+0.5).clamp(0, 1),
                os.path.join(self.output_dir, log_img_filename))

class DistilTargetLogger(VisualisationLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "distil_target"

    def generate_vis_img(self, step, t):
        model_rgb, model_latents = get_outputs(self.model, self.vae)
        noise = torch.randn_like(model_latents)
        noisy_model_latents = self.scheduler.add_noise(model_latents, noise, t)
        noise_pred = predict_noise(self.unet, noisy_model_latents, noise,
                    self.text_embeddings, t, \
                    guidance_scale=self.cfg.guidance_scale,
                    multisteps=self.cfg.multisteps, scheduler=self.scheduler,
                    half_inference=self.cfg.half_inference).clone().detach()
        target_latents = self.scheduler.step(noise_pred, t,
                noisy_model_latents).pred_original_sample\
                        .float().clone().detach()
        if self.cfg.half_inference:
            target_latents = target_latents.half()
        target_rgb = self.vae.decode(target_latents /
            self.vae.config.scaling_factor).sample.float()
        log_img = torch.cat((model_rgb,target_rgb), dim=2)
        return log_img
