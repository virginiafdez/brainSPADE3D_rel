"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models_spade.networks.base_network import BaseNetwork
from models_spade.networks.normalization import get_nonspade_norm_layer
from models_spade.networks.distributions import HypersphericalUniform, VonMisesFisher
import torch
from models_spade.networks.distributions import VonMisesFisher, HypersphericalUniform

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.z_dim = opt.z_dim
        self.layer1 = norm_layer(nn.Conv2d(1, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if min(opt.crop_size) >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
            self.encoder_upsamplings = 6
        else:
            self.encoder_upsamplings = 5

        #self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.wo = w0 = self.opt.crop_size[1] // (2**self.encoder_upsamplings)
        self.ho = h0 = self.opt.crop_size[0] // (2**self.encoder_upsamplings)

        # self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256) # OLD
        # self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256) #OLD
        self.fc_mu = nn.Linear(ndf * 8 * w0 * h0, self.z_dim) # OLD
        if opt.type_prior == 'N':
            self.fc_var = nn.Linear(ndf * 8 * w0 * h0, self.z_dim) #OLD
        elif opt.type_prior == 'S':
            self.fc_var = nn.Linear(ndf * 8 * w0 * h0, 1)

        self.actvn = nn.LeakyReLU(0.2, False)


    def forward(self, x):
        interp = [64 if i < 64 else i for i in x.shape[2:]]
        x = F.interpolate(x, size=tuple(interp), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if min(self.opt.crop_size) >= 256:
            x = self.layer6(self.actvn(x))
        #x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        if self.opt.type_prior == 'N':
            logvar = self.fc_var(x)
        elif self.opt.type_prior == 'S':
            mu = mu / mu.norm(dim=-1, keepdim=True)
            logvar = F.softplus(self.fc_var(x)) + 1

        return mu, logvar

    def reparameterize(self, mu, logvar):

        if self.opt.type_prior == 'N':
            # Normal distribution.
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
        elif self.opt.type_prior == 'S':
            q_z = VonMisesFisher(mu, logvar)
            p_z = HypersphericalUniform(self.opt.z_dim, -1, device=q_z.device)
            z = q_z.rsample()
            return z, q_z, p_z

