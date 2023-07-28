import torch
import monai
from monai.networks.blocks import Convolution, ResidualUnit
import torch.nn as nn
from monai.networks.layers.factories import Act, Norm
import torch.nn.functional as F

class MIA_Decoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels, z_dim, channels, strides, num_res_units,
                 sw, sh, dropout = 0.2,):
        super().__init__()
        self.act = Act.LEAKYRELU
        self.norm = Norm.INSTANCE
        self.k_size = 3
        self.z_dim = z_dim
        self.sh = sh
        self.sw = sw
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.intermediate = torch.nn.Linear(self.z_dim, in_channels*self.sh*self.sw)
        torch.nn.init.xavier_normal_(self.intermediate.weight)
        decode = torch.nn.Sequential()
        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = nn.Sequential()
            conv = Convolution(spatial_dims=2, in_channels=layer_channels, out_channels=c,
                               strides=s, kernel_size=self.k_size, act=self.act,
                               norm=self.norm, dropout= dropout, bias=True,
                               conv_only= i == (len(strides) - 1),
                               is_transposed=True,
            )
            layer.add_module("conv", conv)
            if num_res_units > 0:
                ru = ResidualUnit(
                    spatial_dims=2,
                    in_channels=c,
                    out_channels=c,
                    strides=1,
                    kernel_size=self.k_size,
                    subunits=1,
                    act=self.act,
                    norm=self.norm,
                    dropout=dropout,
                    bias=True,
                    last_conv_only= i == (len(strides) - 1),
                )
                layer.add_module("resunit", ru)
            decode.add_module("decode_%i" % i, layer)
            layer_channels = c

        last_conv = nn.Sequential(*[nn.Conv2d(in_channels=c, out_channels=out_channels,
                                         kernel_size=1, bias = True)])

        layer.add_module("last_conv", last_conv)

        self.decoder = decode

    def forward(self, z):
        x_int = self.intermediate(z)
        x_int = x_int.view(x_int.shape[0], self.in_channels, self.sh, self.sw)
        x_out = self.decoder(x_int)
        x_out = F.leaky_relu(x_out, 0.2)
        return x_out

