from diffusers import StableDiffusionPipeline, DDIMScheduler
from omegaconf import OmegaConf
from utils import SLURM_OUTPUT_DIR
import hydra
import yaml
import os
import matplotlib.pyplot as plt
import torch

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

    generator = torch.Generator(device=device).manual_seed(42)
    pipe_output = pipe(
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        height=cfg.img_resolution, width=cfg.img_resolution,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator
    )
    image = pipe_output.images[0]
    image.save(os.path.join(output_dir,"out.png"))

if __name__ == "__main__":
    main()
