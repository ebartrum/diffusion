import torch
from daam import trace, set_seed
from matplotlib import pyplot as plt
from diffusers import UniDiffuserPipeline

device = "cuda"
model_id_or_path = "thu-ml/unidiffuser-v0"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path)
pipe.to(device)

# Text-to-image generation
prompt = "an elephant under the sea"


# gen = set_seed(0)  # for reproducibility

with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    with trace(pipe) as tc:
        # out = pipe(prompt, num_inference_steps=30, generator=gen)
        sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
        t2i_image = sample.images[0]
        t2i_image.save("unidiffuser_text2img_sample_image.png")
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map('elephant')
        heat_map.plot_overlay(sample.images[0])
        plt.show()
