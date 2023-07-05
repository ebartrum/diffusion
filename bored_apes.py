from diffusers import DDPMPipeline
import matplotlib.pyplot as plt

ddpm = DDPMPipeline.from_pretrained('CCMat/diff-bored-apes-128').to("cuda")
image = ddpm(num_inference_steps=700).images[0]

plt.imshow(image)
plt.show()
