import math
from typing import Callable

import gin
import torch
from torch import Tensor, nn
import tinycudann as tcnn

from neural_field.encoding.tri_mip import TriMipEncoding


@gin.configurable()
class Delta(nn.Module):
    def __init__(
        self,
        encoding_fn,
        n_levels: int = 8,
        plane_size: int = 512,
        feature_dim: int = 16,
        net_depth_base: int = 4,
        net_width: int = 128,
    ) -> None:
        super().__init__()
        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)

        self.encoding_fn = encoding_fn
        self.mlp_base = tcnn.Network(
            n_input_dims=encoding_fn.dim_out+1,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_base,
            },
        )

    def query_delta(
            self, x: Tensor, level_vol: Tensor, t: Tensor
    ):
        level = (
            level_vol if level_vol is None else level_vol + self.log2_plane_size
        )
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        print(x.shape)
        print('before_encoding')
        print(selector.shape)
        with torch.no_grad():
            enc_x = self.encoding_fn(
                x.view(-1, 3),
                level=level.view(-1, 1),
            )
        print('after_encoding')
        enc = torch.concat([enc_x, t], axis=-1) 
        delta = (
            self.mlp_base(enc)
            .view(list(x.shape[:-1]) + [3])
            .to(x)
        )
        print('after_delta')
        print(delta)
        return {"delta": delta}
