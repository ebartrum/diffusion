from diffusers import DDPMPipeline
import matplotlib.pyplot as plt

ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
image = ddpm(num_inference_steps=25).images[0]
image.save("out/image.png")
plt.imshow(image)
plt.show()
