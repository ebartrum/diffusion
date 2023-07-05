import torch
from diffusers import StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# Download images for inpainting example
img_url = "https://www.turing.ac.uk/sites/default/files/styles/people/public/2018-10/edward_bartrum_2.jpg"
init_image = download_image(img_url).resize((512, 512))

# Load the pipeline
device = "cuda"
model_id = "stabilityai/stable-diffusion-2-1-base"
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)

image = img2img_pipe(
    prompt="An oil painting of a man at a beach",
    image = init_image, # The starting image
    strength = 0.6, # 0 for no change, 1.0 for max strength
).images[0]

# Log output
image.save("out/sd_img_2_img.png")
plt.imshow(image)
plt.show()
