import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
image = Image.open("misc/img_2730.png").convert("RGB")
mask_image = Image.open("misc/mask_2730.png").convert("RGB")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32,
)
prompt = "A yellow cat, high resolution"
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("out/inpainted.png")
