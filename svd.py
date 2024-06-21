import torch
from omegaconf import OmegaConf
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
from dataclasses import dataclass

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

@dataclass
class DefaultConfig:
    motion_bucket_id: int = 60
    output_file: str = "generated.mp4"
    image_path: str = "../gaussian-splatting/data/face/images/frame_00044.jpg"

cfg = OmegaConf.merge(OmegaConf.structured(DefaultConfig),
      OmegaConf.from_cli())

# Load the conditioning image
image = Image.open("../gaussian-splatting/data/face/images/frame_00044.jpg")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator,
  motion_bucket_id=cfg.motion_bucket_id).frames[0]

export_to_video(frames, f"out/{cfg.output_file}", fps=7)
