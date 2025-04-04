import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

model_dir, local_files_only = "./models", True
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16",
    cache_dir=model_dir, local_files_only=local_files_only
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("./misc/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
