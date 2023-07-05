import torch
from diffusers import StableDiffusionDepth2ImgPipeline
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# Download images for inpainting example
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
init_image = download_image(img_url).resize((512, 512))

# Load the pipeline
device = "cuda"
model_id = "stabilityai/stable-diffusion-2-depth"
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(model_id).to(device)
prompt = "An oil painting of a man on a bench"
image = pipe(prompt=prompt, image=init_image).images[0]

# Log output
image.save("out/sd_depth2img.png")
plt.imshow(image)
plt.show()
