import torch

from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v0"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path)
pipe.to(device)

# Text-to-image generation
prompt = "an elephant under the sea"

sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]
t2i_image.save("unidiffuser_text2img_sample_image.png")
