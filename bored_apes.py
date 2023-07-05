from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained('CCMat/diff-bored-apes-128')
image = pipeline().images[0]

import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
