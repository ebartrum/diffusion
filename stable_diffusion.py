import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import matplotlib.pyplot as plt

if os.getenv("SLURM_JOB_ID"):
    output_dir = os.path.join("out",
        f"J{os.getenv('SLURM_JOB_ID')}_{os.getenv('SLURM_JOB_NAME')}")
else:
    output_dir = "out"
os.makedirs(output_dir, exist_ok=True)

device = "cuda"
model_id = "stabilityai/stable-diffusion-2-1-base"
model_dir = "./models"
ddim = DDIMScheduler.from_pretrained(
       model_id, subfolder="scheduler",
       cache_dir=model_dir, local_files_only=True)
pipe = StableDiffusionPipeline.from_pretrained(model_id,schedule=ddim,
       cache_dir=model_dir, local_files_only=True).to(device)

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
image.save(os.path.join(output_dir,"sd.png"))
