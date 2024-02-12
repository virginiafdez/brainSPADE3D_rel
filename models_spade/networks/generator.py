"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models_spade.networks.base_network import BaseNetwork
from models_spade.networks.normalization import get_nonspade_norm_layer
from models_spade.networks.architecture import ResnetBlock as ResnetBlock
from models_spade.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from monai.networks.blocks import SubpixelUpsample

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.upsampling_type = opt.upsampling_type

        nf = opt.ngf
        self.sw, self.sh, self.sd = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh * self.sd)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv3d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv3d(final_nc, 1, 3, padding=1)

        if self.upsampling_type == 'upsample' or self.upsampling_type == 'subpixel':
            self.up_b = nn.Upsample(scale_factor=2, mode = 'bicubic')
            self.up_n = nn.Upsample(scale_factor=2, mode = 'nearest')
            if self.upsampling_type == 'subpixel':
                self.sub_pix = SubpixelUpsample(spatial_dims = 2, in_channels=1*nf, out_channels=1*nf,
                                                scale_factor=2,)

        elif self.upsampling_type == 'transposed':
            self.head_0_tc = torch.nn.Sequential(*[torch.nn.ConvTranspose3d(in_channels=16*nf, out_channels=16*nf,
                                                                            kernel_size=3, output_padding=1, stride=2,
                                                                            padding = 1),
                                                   torch.nn.LeakyReLU(0.2)])
            self.G_middle_0_tc = torch.nn.Sequential(*[torch.nn.ConvTranspose3d(in_channels=16*nf, out_channels=16*nf,
                                                                            kernel_size=3, output_padding=1, stride=2,
                                                                                padding = 1),
                                                   torch.nn.LeakyReLU(0.2)])
            self.G_middle_1_tc = torch.nn.Sequential(*[torch.nn.ConvTranspose3d(in_channels=16*nf, out_channels=16*nf,
                                                                            kernel_size=3, output_padding=1, stride=2,
                                                                                padding = 1),
                                                   torch.nn.LeakyReLU(0.2)])
            self.up_0_tc = torch.nn.Sequential(*[torch.nn.ConvTranspose3d(in_channels=8*nf, out_channels=8*nf,
                                                                            kernel_size=3, output_padding=1, stride=2,
                                                                          padding = 1),
                                                   torch.nn.LeakyReLU(0.2)])
            self.up_1_tc = torch.nn.Sequential(*[torch.nn.ConvTranspose3d(in_channels=4*nf, out_channels=4*nf,
                                                                            kernel_size=3, output_padding=1, stride=2,
                                                                          padding = 1),
                                                   torch.nn.LeakyReLU(0.2)])
            self.up_2_tc = torch.nn.Sequential(*[torch.nn.ConvTranspose3d(in_channels=2*nf, out_channels=2*nf,
                                                                            kernel_size=3, output_padding=1, stride=2,
                                                                          padding = 1),
                                                   torch.nn.LeakyReLU(0.2)])
            self.up_3_tc = torch.nn.Sequential(*[torch.nn.ConvTranspose3d(in_channels=1*nf, out_channels=1*nf,
                                                                            kernel_size=3, output_padding=1, stride=2,
                                                                          padding = 1),
                                                   torch.nn.LeakyReLU(0.2)])
        else:
            ValueError("Unrecognized upsampling type. Types allowed: upsample and transposed. Entered %s" %self.upsampling_type)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        #opt.chunk_size = None
        if opt.chunk_size is None:
            chunk_size = opt.crop_size[-1]
        else:
            chunk_size = opt.chunk_size

        sd = chunk_size // (2**num_up_layers) #opt.crop_size[2] // (2**num_up_layers)
        sw = opt.crop_size[1] // (2**num_up_layers)
        sh = opt.crop_size[0] // (2**num_up_layers)

        #sh = round(sw / opt.aspect_ratio)

        return sw, sh, sd

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw, self.sd)
        else:
            # we downsample segmap and run convolution

            x = F.interpolate(seg, size=(self.sh, self.sw, self.sd))
            x = self.fc(x)

        x = self.head_0(x, seg)

        if self.upsampling_type == 'transposed':
            x = self.head_0_tc(x)
        else:
            x = self.up_n(x)

        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            if self.upsampling_type == 'transposed':
                x = self.G_middle_0_tc(x)
            else:
                x = self.up_n(x)

        x = self.G_middle_1(x, seg)
        if self.upsampling_type == 'transposed':
            x = self.G_middle_1_tc(x)
        else:
            x = self.up_n(x)
        x = self.up_0(x, seg)
        if self.upsampling_type == 'transposed':
            x = self.up_0_tc(x)
        else:
            x = self.up_n(x)
        x = self.up_1(x, seg)
        if self.upsampling_type == 'transposed':
            x = self.up_1_tc(x)
        else:
            x = self.up_n(x)
        x = self.up_2(x, seg)
        if self.upsampling_type == 'transposed':
            x = self.up_2_tc(x)
        else:
            x = self.up_n(x)
        x = self.up_3(x, seg)
        if self.opt.num_upsampling_layers == 'most':
            if self.upsampling_type == 'transposed':
                x = self.up_3_tc(x)
            elif self.upsampling_type == 'subpixel':
                x = self.sub_pix(x)
            else:
                x = self.up_n(x)
            x = self.up_4(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.leaky_relu(x, 0.2)

        return x

class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad3d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv3d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv3d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose3d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad3d(3),
                  nn.Conv3d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
