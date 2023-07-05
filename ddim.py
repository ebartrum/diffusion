from diffusers import DDIMScheduler, DDPMPipeline
from PIL import Image
import matplotlib.pyplot as plt

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to("cuda")
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=40)

image_pipe.scheduler = scheduler
images = image_pipe(num_inference_steps=40).images
out = images[0]

plt.imshow(out)
plt.show()
