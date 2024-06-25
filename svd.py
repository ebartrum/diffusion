import torch
from torchvision.io import write_video
from omegaconf import OmegaConf
from diffusers import StableVideoDiffusionPipeline
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
    num_inference_steps: int = 25
    output_file: str = "multi_video.mp4"
    image_path: str = "../gaussian-splatting/data/face/images/frame_00044.jpg"

cfg = OmegaConf.merge(OmegaConf.structured(DefaultConfig),
      OmegaConf.from_cli())

# Load the conditioning images
img1 = Image.open("../gaussian-splatting/data/face/images/frame_00044.jpg")
img1 = img1.resize((1024, 576))
img2 = Image.open("../gaussian-splatting/data/face/images/frame_00045.jpg")
img2 = img2.resize((1024, 576))

flow1 = torch.load("data/flows_44_to_45.pt").cuda()
flow2 = torch.load("data/flows_45_to_44.pt").cuda()

generator = torch.manual_seed(42)
frames = new_call(pipe, [img1,img2], decode_chunk_size=8,
  generator=generator, num_inference_steps=cfg.num_inference_steps,
  motion_bucket_id=cfg.motion_bucket_id,
  flow1=flow1, flow2=flow2, output_type='pt', return_dict=False)

combined_frames = torch.cat([frames[0],frames[1]], dim=2)
combined_frames = (combined_frames*255).to(torch.uint8).cpu()
write_video(f"out/{cfg.output_file}", combined_frames.permute(0,2,3,1), fps=7)
