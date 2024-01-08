import torch
from PIL import Image
from submodules.zoedepth.zoedepth.utils.misc import (colorize,
    pil_to_batched_tensor, save_raw_16bit, get_image_from_url)

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Local file
image = Image.open("misc/rocket.png").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy
depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image
depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

# Tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)
depth = zoe.infer_pil(image)

# Save raw
fpath = "out/zoe_output.png"
save_raw_16bit(depth, fpath)

colored = colorize(depth, cmap="magma_r")
fpath_colored = "out/zoe_output_colored.png"
Image.fromarray(colored).save(fpath_colored)
