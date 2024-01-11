import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import hydra

@hydra.main(config_path="conf/inpainting",
            config_name="config", version_base=None)
def main(cfg):
    image = Image.open(cfg.data.img).convert("RGB")
    mask_image = Image.open(cfg.data.mask).convert("RGB")

    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     # "runwayml/stable-diffusion-inpainting",
    #     "stabilityai/stable-diffusion-2-inpainting",
    #     torch_dtype=torch.float32,
    # )
    pipe = AutoPipelineForInpainting.from_pretrained(
          "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
          torch_dtype=torch.float16,
          variant="fp16").to("cuda")

    image = pipe(prompt=cfg.prompt, image=image, mask_image=mask_image,
                 guidance_scale=cfg.guidance_scale).images[0]
    image.save("out/inpainted.png")

if __name__ == "__main__":
    main()
