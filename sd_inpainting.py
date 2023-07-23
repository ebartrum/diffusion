import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import argparse

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_image(filename, size=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img

def refine_mask(mask, dilation=2):
    res = mask.shape[-1]
    # binarise
    mask = (mask>0.5).float()
    # dilate
    mask = F.max_pool2d(mask.unsqueeze(0),dilation).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
         res).squeeze(0).squeeze(0)
    return mask

parser = argparse.ArgumentParser()
parser.add_argument('--input_img', '-i', type=str, required=True)           # positional argument
parser.add_argument('--input_mask', '-m', type=str, required=True)           # positional argument
parser.add_argument('--prompt', '-p', type=str, required=True)           # positional argument
parser.add_argument('--gpu', '-g', type=int, default=0)           # positional argument
args = parser.parse_args()

init_image = load_image(args.input_img).resize((512, 512))
mask_image = load_image(args.input_mask).resize((512, 512))
init_image = torch.from_numpy(np.array(init_image)).permute(2,0,1)/255
mask_image = torch.from_numpy(np.array(mask_image)).permute(2,0,1)[0]/255

mask_image = refine_mask(mask_image, dilation=32)
init_image = 2*init_image - 1

# Load the pipeline
device = f"cuda:{args.gpu}"
model_id = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id).to(device)
with open(args.prompt, "r") as f:
    prompt = f.readline()
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

# Log output
image.save("runs/sd_inpainting.png")
image_tensor = torch.from_numpy(np.array(image))
masked_init_img = init_image*(1-mask_image).unsqueeze(0)
masked_init_img = 0.5*masked_init_img.permute(1,2,0).float() + 0.5
vis_init_img = 0.5*init_image.permute(1,2,0).float() + 0.5

combined = torch.cat([vis_init_img, masked_init_img,
      image_tensor.float()/255], dim=1)
save_image(combined.permute(2,0,1),"runs/inpainting_combined.png")
