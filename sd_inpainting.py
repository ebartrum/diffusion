import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting

image = Image.open("misc/rednet_img.png").convert("RGB")
mask_image = Image.open("misc/rednet_mask.png").convert("RGB")

# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     # "runwayml/stable-diffusion-inpainting",
#     "stabilityai/stable-diffusion-2-inpainting",
#     torch_dtype=torch.float32,
# )
pipe = AutoPipelineForInpainting.from_pretrained(
      "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
      torch_dtype=torch.float16,
      variant="fp16").to("cuda")

# prompt = "empty plinth"
prompt = ""
image = pipe(prompt=prompt, image=image, mask_image=mask_image,
             guidance_scale=7.5).images[0]
image.save("out/inpainted.png")
