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

    def generate(self, deformation_code=None, step=None):
        return self.output_tensor

    def parameter_groups(self, lr):
        return self.parameters()

class LatentTensor(nn.Module):
    def __init__(self, img_res=512):
        super().__init__()
        self.distillation_space = "latent"
        self.output_tensor = torch.nn.parameter.Parameter(
                torch.randn((4, img_res // 8, img_res // 8)))

    def generate(self, deformation_code=None, step=None):
        return self.output_tensor

    def parameter_groups(self, lr):
        return self.parameters()

class InstantNGP(nn.Module):
    def __init__(self, img_res=512, distillation_space="rgb",
             noise_anneal_steps=200):
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
        self.output_size = img_res // 8 if distillation_space=="latent" else img_res
        resolution = img_res, img_res
        half_dx =  0.5 / self.output_size
        half_dy =  0.5 / self.output_size
        xs = torch.linspace(half_dx, 1-half_dx, self.output_size)
        ys = torch.linspace(half_dy, 1-half_dy, self.output_size)
        xv, yv = torch.meshgrid([xs, ys])
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        self.xy = torch.nn.parameter.Parameter(xy, requires_grad=False)
        self.noise_anneal_steps = noise_anneal_steps

    def parameter_groups(self, lr):
        return [{'params': self.encoding.parameters(), 'lr': 10*lr},
                {'params': self.network.parameters(), 'lr': lr}]

    def apply_noise_annealing(self, out, step):
        noise = torch.randn_like(out)
        noise_level = torch.clip(
                torch.tensor(1-(step/self.noise_anneal_steps)), 0, 1)
        out = noise_level*noise + (1-noise_level)*out
        return out

    def generate(self, deformation_code=None, step=None):
        out = self.net(self.xy).resize(self.output_size,self.output_size,
               self.out_features).permute(2,0,1)
        if step is not None:
            out = self.apply_noise_annealing(out, step)
        return out

class DeformableInstantNGP(InstantNGP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_deformations = torch.nn.parameter.Parameter(
                torch.cat([self.xy, 1-self.xy], 1).min(1).values,
                requires_grad=False)
        deformation_encoding_cfg = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 16,
            "per_level_scale": 1.5
            }
        deformation_network_cfg = {
           "otype": "FullyFusedMLP",
           "activation": "Sine",
           "output_activation": "None",
           "n_neurons": 32,
           "n_hidden_layers": 1}
        self.deformation_encoding = tcnn.Encoding(
            n_input_dims=2, encoding_config=deformation_encoding_cfg)
        self.deformation_network = tcnn.Network(
                n_input_dims=self.deformation_encoding.n_output_dims,
               n_output_dims=2,
               network_config=deformation_network_cfg)
        self.deformation_net = torch.nn.Sequential(self.deformation_encoding,
                   self.deformation_network)

    def parameter_groups(self, lr):
        return [{'params': self.encoding.parameters(), 'lr': 10*lr},
                {'params': self.deformation_encoding.parameters(), 'lr': 10*lr},
                {'params': self.network.parameters(), 'lr': lr},
                {'params': self.deformation_network.parameters(), 'lr': lr},
                ]

    def deformed_xy(self, deformation_code=None, step=None):
        deformation_linear = self.deformation_net(self.xy)
        max_amplitude = self.max_deformations
        if step is not None:
            max_amplitude = max_amplitude * (torch.tensor(step)/200).clip(0,1)
        deformation_offset = torch.tanh(deformation_linear) * max_amplitude.unsqueeze(1)
        return self.xy + deformation_offset

    def generate(self, deformation_code=None, step=None):
        deformed_xy = self.deformed_xy(deformation_code, step)
        out = self.net(deformed_xy).resize(self.output_size,self.output_size,
               self.out_features).permute(2,0,1)
        if step is not None:
            out = self.apply_noise_annealing(out, step)
        return out
