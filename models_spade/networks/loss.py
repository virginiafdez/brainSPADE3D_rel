"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import warnings

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

# Morning <3

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None, drop_first = False):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        self.drop_first = drop_first
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input).type(input.type())

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    # If the target is negative
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multi-scale discriminator
        if isinstance(input, list):
            loss = 0
            for ind, pred_i in enumerate(input):
                if ind == 0 and self.drop_first:
                    # With drop first, we leave the output of the first discriminator out
                    # because the receptive field is too small.
                    continue
                else:
                    if isinstance(pred_i, list):
                        pred_i = pred_i[-1] # We keep the last of the tensors in the list, which is the output of the discriminator
                    loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                    loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class KLDLossVMF(nn.Module):
    def forward(self, q_z, p_z):
        return torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()

class latentModalityLoss(nn.Module):
    def __init__(self, latent_space_size, modalities, mu = 0.5, margin = 0.003):
        """
        Creates a Modality Loss function in Latent space.
        :param latent_space_size: Size of the latent space (number of dimensions)
        :param classes, list, name of the classes corresponding to the modalties
        :param mu, float, weight given to the anchor-positive loss with regards to the anchor positive loss
        :param margin. The Triplet Loss function minimises the difference between the anchor-positive
        (L2 norm between samples labelled the same) and anchor-negative (L2 norm between samples with different
        labels) + m, where m is the margin, the perimeter we want to preserve between the samples with the same
        label and the impostors. Default: 0.003 (given the fact that our autoencoder is attempting to achieve
        a N (0, 1) distribution, with an overall width of 3STD = 3.
        """
        super(latentModalityLoss, self).__init__()
        self.latent_space_size = latent_space_size
        self.mu = mu
        self.modalities = modalities
        self.margin = margin

    def forward(self, batch_codes, modalities, epsilon = 0.0001):

        """
        :param batch_codes:
        :param modalities: list of the modality associated to each batch item.
        Must match those mapped in self.classes
        :param epsilon:
        :return:
        """

        # Filter warnings
        warnings.filterwarnings("ignore")

        # We store all codes pertaining to one class in a
        mod_dict = {}
        for mod in self.modalities:
            filter = [True  if m_==mod else False for m_ in modalities]
            mod_dict[mod] = batch_codes[filter]

        loss = 0
        n_it = 0

        for key, value in mod_dict.items():
            n_items = value.shape[0]
            pairs = list(itertools.combinations(range(n_items), 2))
            for key_impostor, value_impostor in mod_dict.items():
                if key_impostor == key:
                    continue
                else:
                    for impostor_ in range(value_impostor.shape[0]):
                        for pair in pairs:
                            # Combinations doesn't care about order (0,1) = (1,0)
                            # To take into account the distances of both elements of the pair
                            # and the impostor, we add the loss both-ways to the global loss.
                            l2_targetN = self.l2_norm(value[pair[0]], value[pair[1]])
                            loss += max(l2_targetN
                                        - self.l2_norm(value[pair[0]], value_impostor[impostor_])
                                        + self.margin, 0)
                            loss += max(l2_targetN
                                        - self.l2_norm(value[pair[1]], value_impostor[impostor_])
                                        + self.margin, 0)
                            n_it += 2

        loss = loss / n_it
        return torch.tensor(loss, requires_grad=True).cuda()

    def l2_norm(self, code1, code2):
        return torch.sqrt(torch.pow(code1-code2, 2).sum())

    def l1_norm(self, code1, code2):
        return torch.abs((code1-code2).sum())

def l2_norm(code1, code2):
    return torch.sqrt(torch.pow(code1 - code2, 2).sum())

def l1_norm(code1, code2):
    return torch.abs((code1 - code2).sum())

def coSim(code1, code2):
    return torch.nn.functional.cosine_similarity(code1, code2, 0)

def coSimI(code1, code2):
    return 1.0 - torch.nn.functional.cosine_similarity(code1, code2, -1)