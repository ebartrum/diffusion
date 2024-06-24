import torch
from omegaconf import OmegaConf
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
from dataclasses import dataclass
from svd_logic_hacking import new_call, new_step

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

@dataclass
class DefaultConfig:
    motion_bucket_id: int = 60
    output_file0: str = "generated0.mp4"
    output_file1: str = "generated1.mp4"
    image_path: str = "../gaussian-splatting/data/face/images/frame_00044.jpg"

cfg = OmegaConf.merge(OmegaConf.structured(DefaultConfig),
      OmegaConf.from_cli())

# Load the conditioning images
img1 = Image.open("../gaussian-splatting/data/face/images/frame_00044.jpg")
img1 = img1.resize((1024, 576))
img2 = Image.open("../gaussian-splatting/data/face/images/frame_00045.jpg")
img2 = img2.resize((1024, 576))

generator = torch.manual_seed(42)
frames = new_call(pipe, [img1,img2], decode_chunk_size=8, generator=generator,
  motion_bucket_id=cfg.motion_bucket_id).frames

print(f"Saving video 0 to out/{cfg.output_file0} ...")
export_to_video(frames[0], f"out/{cfg.output_file0}", fps=7)
print(f"Saving video 1 to out/{cfg.output_file1} ...")
export_to_video(frames[1], f"out/{cfg.output_file1}", fps=7)
