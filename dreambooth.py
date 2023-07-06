from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

dataset_id = "lewtun/corgi"  # CHANGE THIS TO YOUR {hub_username}/{dataset_id}
dataset = load_dataset(dataset_id, split="train")

num_samples = 4
input_grid = image_grid(dataset["image"][:num_samples], rows=1, cols=num_samples)

plt.imshow(input_grid)
#plt.show()

name_of_your_concept = "ccorgi"  # CHANGE THIS ACCORDING TO YOUR SUBJECT
type_of_thing = "dog"  # CHANGE THIS ACCORDING TO YOUR SUBJECT
instance_prompt = f"a photo of {name_of_your_concept} {type_of_thing}"
print(f"Instance prompt: {instance_prompt}")
