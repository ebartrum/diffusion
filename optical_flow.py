import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.io import read_video
from torchvision.io import read_image
from torchvision.utils import save_image
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import raft_large
import os

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

img1_path = "/home/ed/Documents/data/nerfstudio/face/images/frame_00044.jpg"
img2_path = "/home/ed/Documents/data/nerfstudio/face/images/frame_00045.jpg"
img1_batch = read_image(img1_path).unsqueeze(0)
img2_batch = read_image(img2_path).unsqueeze(0)

# If you can, run this example on a GPU, it will be a lot faster.
device = "cuda" if torch.cuda.is_available() else "cpu"

img1_batch = preprocess(img1_batch).to(device)
img2_batch = preprocess(img2_batch).to(device)

print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")

predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img + 1) / 2 for img in img1_batch]
img2_batch = [(img + 1) / 2 for img in img2_batch]

def apply_warp(img, flow):
    grid_x, grid_y = torch.meshgrid(torch.tensor(range(img.shape[1])),
        torch.tensor(range(img.shape[2])), indexing='ij')
    identity_flow = torch.stack([grid_y,grid_x]).unsqueeze(0).to(img.device).float()
    flow = identity_flow - flow #switch from pixel offsets to absolute positions

    #Normalize flow
    flow[:,0] /= img.shape[2]
    flow[:,1] /= img.shape[1]

    #set flow to range [-1,1]
    flow = flow*2 - 1

    flow_permute = torch.permute(flow, (0, 2, 3, 1))
    remapped = torch.nn.functional.grid_sample(img.unsqueeze(0), flow_permute)
    return remapped

img1_warped = apply_warp(img1_batch[0], predicted_flows)
img1_warped = [img1_warped.squeeze(0)]

save_image(img1_batch, "out/optical_flow/img1.png")
save_image(img1_warped, "out/optical_flow/img1_warped.png")
save_image(img2_batch, "out/optical_flow/img2.png")

flow_filename = f"flow_{os.path.basename(img1_path).replace('.jpg','')}_{os.path.basename(img2_path).replace('.jpg','')}.pt"
torch.save(predicted_flows, f"out/optical_flow/{flow_filename}")
