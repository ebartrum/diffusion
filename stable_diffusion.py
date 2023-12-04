import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os

device = "cuda"
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

generator = torch.Generator(device=device).manual_seed(42)
pipe_output = pipe(
    prompt="A cute and realistic kitten",
    negative_prompt="Oversaturated, blurry, low quality",
    height=512, width=512,
    guidance_scale=8,
    num_inference_steps=35,
    generator=generator
)

image = pipe_output.images[0]

# Log output
output_dir = "runs"
image.save(os.path.join("out","sd.png"))
plt.imshow(image)
plt.show()
