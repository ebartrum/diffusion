import torch
from tqdm import tqdm
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
import numpy as np
import matplotlib.pyplot as plt

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(1000)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).cuda()
input = noise

for t in tqdm(scheduler.timesteps):
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))

plt.imshow(image)
plt.show()
