from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import torch

device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipe = pipe.to(device)

prompt = 'A dog runs across the field'
gen = set_seed(0)  # for reproducibility

with torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=30, generator=gen)
        plt.imshow(out.images[0])
        plt.show()
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map('dog').heatmap
        plt.imshow(heat_map.cpu())
        plt.show()
