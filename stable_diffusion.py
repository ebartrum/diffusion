import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Load the pipeline
device = "cuda"
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

generator = torch.Generator(device=device).manual_seed(42)
# Run the pipeline, showing some of the available arguments
pipe_output = pipe(
    prompt="Palette knife painting of 2 cats playing chess", # What to generate
    negative_prompt="Oversaturated, blurry, low quality", # What NOT to generate
    height=480, width=480,     # Specify the image size
    guidance_scale=8,          # How strongly to follow the prompt
    num_inference_steps=35,    # How many steps to take
    generator=generator        # Fixed random seed
)

# View the resulting image:
image = pipe_output.images[0]

# Log output
image.save("out/sd.png")
plt.imshow(image)
plt.show()
