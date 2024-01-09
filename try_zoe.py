import os
import torch
from PIL import Image
from submodules.zoedepth.zoedepth.utils.misc import (colorize,
    pil_to_batched_tensor, save_raw_16bit, get_image_from_url)
from submodules.zoedepth.zoedepth.utils.config import get_config
from submodules.zoedepth.zoedepth.models.builder import build_model

if os.getenv("SLURM_JOB_ID"):
    output_dir = os.path.join("out", SLURM_OUTPUT_DIR)
else:
    output_dir = "out"
os.makedirs(output_dir, exist_ok=True)

conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Local file
image = Image.open("misc/corgi.png").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy
depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image
depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

# Tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)
depth = zoe.infer_pil(image)

# Save raw
fpath = os.path.join(output_dir, "zoe_output.png")
grey_depth = colorize(depth, cmap="gray_r")
Image.fromarray(grey_depth).save(fpath)

colored = colorize(depth, cmap="magma_r")
fpath_colored = os.path.join(output_dir, "zoe_output_colored.png")
Image.fromarray(colored).save(fpath_colored)
