from diffusers import DDPMPipeline
import matplotlib.pyplot as plt

pipeline = DDPMPipeline.from_pretrained('CCMat/diff-bored-apes-128').to("cuda")
image = pipeline().images[0]

plt.imshow(image)
plt.show()
