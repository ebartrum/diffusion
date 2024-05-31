import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import to_tensor, center_crop
import requests
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMScheduler

@torch.no_grad()
def invert(
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cpu",
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)

device = torch.device("cuda:0")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

img = Image.open("./data/frontal_face.jpg").convert("RGB")
img = to_tensor(img).to(device)
img = center_crop(img,min(img.shape[1],img.shape[2]))
img = F.interpolate(img.unsqueeze(0),512).squeeze(0)
input_image_prompt = "A man"

# Encode with VAE
with torch.no_grad():
    latent = pipe.vae.encode(img.unsqueeze(0)*2 - 1)

l = 0.18215 * latent.latent_dist.sample()

cfg = 1
num_steps = 999

inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_steps,
      guidance_scale=cfg,
      device=device)

final_inverted_latent = inverted_latents[-1].unsqueeze(0)

reconstruction = pipe(input_image_prompt, latents=final_inverted_latent,
      num_inference_steps=num_steps, guidance_scale=cfg, eta=0,
      output_type="pt").images[0]

combined_output = torch.cat((img,reconstruction),2)
torchvision.utils.save_image(combined_output,"out/inversion.png")
plt.imshow(combined_output.cpu().permute(1,2,0))
plt.show()
