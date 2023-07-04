import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]

image.save("out/image.png")
plt.imshow(image)
plt.show()
