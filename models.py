import torch
import torch.nn as nn

class RGBTensor(nn.Module):
    def __init__(self, img_res=512):
        super().__init__()
        self.distillation_space = "rgb"
        self.output_tensor = torch.nn.parameter.Parameter(
                torch.randn((3, img_res, img_res)))

    def generate(self):
        return self.output_tensor

class LatentTensor(nn.Module):
    def __init__(self, img_res=512):
        super().__init__()
        self.distillation_space = "latent"
        self.output_tensor = torch.nn.parameter.Parameter(
                torch.randn((4, img_res // 8, img_res // 8)))

    def generate(self):
        return self.output_tensor
