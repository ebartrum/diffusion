import os
import torch
from torchvision.utils import save_image

class VisualisationLogger():
    def __init__(self, output_dir, log_steps=50, total_steps=1000):
        self.name = "img_log"
        self.log_steps = log_steps
        self.total_steps = total_steps
        self.output_dir = output_dir

    def is_log_step(self, step):
        out = self.log_steps and (step % self.log_steps == 0 or
                step == (self.total_steps-1))
        return out

    def log_img(self, step):
        log_img = self.generate_vis_img(step)
        log_img_filename = f"{self.name}_step.png"
        save_image((log_img/2+0.5).clamp(0, 1),
                os.path.join(self.output_dir, log_img_filename))

class DistilTargetLogger(VisualisationLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "distil_target"

    def generate_vis_img(self, step):
        return torch.ones(3,128,128)
