import torch
import torch.nn as nn
import json
import tinycudann as tcnn

class RGBTensor(nn.Module):
    def __init__(self, img_res=512):
        super().__init__()
        self.distillation_space = "rgb"
        self.output_tensor = torch.nn.parameter.Parameter(
                torch.randn((3, img_res, img_res)))

    def generate(self, deformation_code=None):
        return self.output_tensor

    def parameter_groups(self, lr):
        return self.parameters()

class LatentTensor(nn.Module):
    def __init__(self, img_res=512):
        super().__init__()
        self.distillation_space = "latent"
        self.output_tensor = torch.nn.parameter.Parameter(
                torch.randn((4, img_res // 8, img_res // 8)))

    def generate(self, deformation_code=None):
        return self.output_tensor

    def parameter_groups(self, lr):
        return self.parameters()

class InstantNGP(nn.Module):
    def __init__(self, img_res=512, distillation_space="rgb"):
        super().__init__()
        self.distillation_space = distillation_space
        self.out_features = 4 if distillation_space=="latent" else 3
        config_path = "conf/config_hash.json"
        with open(config_path) as config_file:
            config = json.load(config_file)
        self.encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
        self.network = tcnn.Network(n_input_dims=self.encoding.n_output_dims,
               n_output_dims=self.out_features, network_config=config["network"])
        self.net = torch.nn.Sequential(self.encoding, self.network)
        self.output_size = img_res // 8 if distillation_space=="rgb" else img_res
        resolution = img_res, img_res
        half_dx =  0.5 / self.output_size
        half_dy =  0.5 / self.output_size
        xs = torch.linspace(half_dx, 1-half_dx, self.output_size)
        ys = torch.linspace(half_dy, 1-half_dy, self.output_size)
        xv, yv = torch.meshgrid([xs, ys])
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        self.xy = torch.nn.parameter.Parameter(xy, requires_grad=False)

    def parameter_groups(self, lr):
        return [{'params': self.encoding.parameters(), 'lr': 10*lr},
                {'params': self.network.parameters(), 'lr': lr}]

    def generate(self, prompt_index=0):
        out = self.net(self.xy).resize(self.output_size,self.output_size,
               self.out_features).permute(2,0,1)
        return out

class DeformableInstantNGP(InstantNGP):
    def __init__(self, img_res=512, distillation_space="rgb"):
        super().__init__(img_res, distillation_space)
        self.deformation_net = None

    def parameter_groups(self, lr):
        return [{'params': self.encoding.parameters(), 'lr': 10*lr},
                {'params': self.network.parameters(), 'lr': lr}]

    def deformed_xy(self, deformation_code=None):
        return self.xy
        # import ipdb;ipdb.set_trace()

    def generate(self, deformation_code=None):
        deformed_xy = self.deformed_xy(deformation_code)
        out = self.net(deformed_xy).resize(self.output_size,self.output_size,
               self.out_features).permute(2,0,1)
        return out
