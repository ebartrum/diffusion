import os
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from daam import trace
from datetime import datetime

# Useful function for later
def load_imageurl(url, size=None):
    response = requests.get(url,timeout=0.2)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img

def load_image(filename, size=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img

device = torch.device("cuda")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Sample an image to make sure it is all working
if False:
    prompt = 'Beautiful DSLR Photograph of a penguin on the beach, golden hour'
    negative_prompt = 'blurry, ugly, stock photo'
    im = pipe(prompt, negative_prompt=negative_prompt).images[0]
    plt.imshow(im); plt.show()

input_image = load_imageurl('https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg', size=(512, 512))
input_image_prompt = "Photograph of a puppy on the grass"

# encode with VAE
with torch.no_grad():
    latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)*2-1)
l = 0.18215 * latent.latent_dist.sample()

# Sample function (regular DDIM)
@torch.no_grad()
def sample(prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images

## Inversion
@torch.no_grad()
def invert(start_latents, prompt, guidance_scale=3.5, num_inference_steps=80,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1: continue

        t = timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000//num_inference_steps))#t
        next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred


        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)

if False:
    inverted_latents = invert(l, input_image_prompt,num_inference_steps=50)

if False:
    inverted_img = pipe(input_image_prompt,
        latents=inverted_latents[-1][None], num_inference_steps=50, guidance_scale=3.5).images[0]
    plt.imshow(inverted_img); plt.show()

# We want to be able to specify start step
if False:
    start_step=20
    inverted_img = sample(input_image_prompt, start_latents=inverted_latents[-(start_step+1)][None],
           start_step=start_step, num_inference_steps=50)[0]
    plt.imshow(inverted_img); plt.show()

# Sampling with a new prompt
if False:
    start_step=10
    new_prompt = input_image_prompt.replace('puppy', 'cat')
    new_prompt_img = sample(new_prompt, start_latents=inverted_latents[-(start_step+1)][None],
           start_step=start_step, num_inference_steps=50)[0]
    plt.imshow(new_prompt_img); plt.show()

def edit(input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5):
    with torch.no_grad(): latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)*2-1)
    l = 0.18215 * latent.latent_dist.sample()
    inverted_latents = invert(l, input_image_prompt,num_inference_steps=num_steps)
    final_im = sample(edit_prompt, start_latents=inverted_latents[-(start_step+1)][None],
                      start_step=start_step, num_inference_steps=num_steps, guidance_scale=guidance_scale)[0]
    return final_im

input_img = load_image('data/room_above.png', size=(512, 512))
input_prompt = 'A photograph of a meeting room with a screen and cables on the table'
edit_prompt = 'A photograph of a meeting room with a screen and nothing on the table'
heatmap_word = 'nothing'

with trace(pipe) as tc:
    edited_img = edit(input_img, input_prompt,
       edit_prompt, num_steps=30,
       start_step=20, guidance_scale=3.5)
    edited_img.show()
    heat_map = tc.compute_global_heat_map()
    heat_map = heat_map.compute_word_heat_map(heatmap_word).heatmap
    plt.imshow(heat_map.cpu())
    plt.show()

today = datetime.now()
current_time = f"{today.year}-{today.month}-{today.day}_{today.hour}:{today.minute}:{today.second}"
run_dir = os.path.join("runs",current_time)
os.makedirs(run_dir, exist_ok=True)

heat_map = F.interpolate(heat_map.unsqueeze(0).unsqueeze(0), input_image.size[0]).squeeze(0)
input_img.save(os.path.join(run_dir, "input.png"))
edited_img.save(os.path.join(run_dir, "edit.png"))

#Normalise
heat_map -= heat_map.min()
heat_map /= heat_map.max()
save_image(heat_map, os.path.join(run_dir, "heatmap.png"))
