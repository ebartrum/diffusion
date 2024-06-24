import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.io import read_video
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import raft_large

plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2

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

video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
_ = urlretrieve(video_url, video_path)

frames, _, _ = read_video(str(video_path))
frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

img1_batch = torch.stack([frames[100]])
img2_batch = torch.stack([frames[101]])
# plot(img1_batch)

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

# import ipdb;ipdb.set_trace()

# flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img + 1) / 2 for img in img1_batch]
img2_batch = [(img + 1) / 2 for img in img2_batch]

def apply_warp(img, flow):
    grid_x, grid_y = torch.meshgrid(torch.tensor(range(img.shape[1])),
        torch.tensor(range(img.shape[2])), indexing='ij')
    identity_flow = torch.stack([grid_x,grid_y]).unsqueeze(0).to(img.device).float()
    flow = identity_flow #temporarily switch to identity flow
    flow[:,0] /= 520
    flow[:,1] /= 960
    # import ipdb;ipdb.set_trace()

    flow_permute = torch.permute(flow, (0, 2, 3, 1))
    remapped = torch.nn.functional.grid_sample(img.unsqueeze(0), flow_permute)
    return remapped

img1_warped = apply_warp(img1_batch[0], predicted_flows)
img1_warped = [img1_warped.squeeze(0)]

grid = [img1_warped, img2_batch]
# grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
plot(grid)

plt.show()
