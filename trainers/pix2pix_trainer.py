"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# test
from models_spade.networks.sync_batchnorm import DataParallelWithCallback
from models_spade.pix2pix_model import Pix2PixModel
import torch
import utils.util as util
import numpy as np
import shutil
import monai
import os
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from models_spade.networks.perceptual_loss_monaigen import PerceptualLoss
from torch.distributed.elastic.multiprocessing.errors import  record
@record
class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, device):
        self.opt = opt
        self.use_ddp = opt.use_ddp
        self.use_dp = opt.use_dp
        self.device = device
        if not self.opt.no_perceptual_loss and self.opt.isTrain:
            perceptual_loss = PerceptualLoss(is_fake_3d=False, spatial_dims=3,
                                             network_type="medicalnet_resnet10_23datasets",
                                             from_preloaded=False,
                                             path_net=self.opt.perceptual_loss_path,
                                             device = self.device)
            for p in perceptual_loss.perceptual_function.model.parameters():
                p.requires_grad = False
            perceptual_loss = perceptual_loss.to(device)
        else:
            perceptual_loss = None
        self.pix2pix_model = Pix2PixModel(opt, device, perceptual_loss).to(device)
        if self.use_ddp:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model
            # ranks = list(range(dist.get_world_size()))
            # r1, r2 =  ranks[:dist.get_world_size()//2], ranks[dist.get_world_size()//2:]
            #Note: every rank calls into new_group for every  process group created, even if that rank is not
            #part of the group (as per https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm)
            # process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            # process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            # self.pix2pix_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.pix2pix_model, process_group)
            self.pix2pix_model = DistributedDataParallel(self.pix2pix_model,
                                                         device_ids=[self.device],
                                                         find_unused_parameters=True,
                                                         broadcast_buffers=False)
        elif self.use_dp:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model, device_ids = opt.gpu_ids)
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        self.g_losses = None
        self.g_losses_noweight = None
        self.d_losses = None
        self.d_accuracy = None
        if opt.isTrain:
            self.old_lr = opt.lr
            self.disc_threshold = {'low': opt.disc_acc_lowerth, 'up': opt.disc_acc_upperth}
            self.optimizer_G, self.optimizer_D = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            if self.opt.continue_train:
                self.loadOptimizers()
        self.batch_accumulation = {}
        self.batch_accumulation['curr_D'] = 0
        self.batch_accumulation['curr_G'] = 0
        if opt.batch_acc_fr > 0:
            self.batch_accumulation['on'] = True
            self.batch_accumulation['freq'] = opt.batch_acc_fr
        else:
            self.batch_accumulation['on'] = False
            self.batch_accumulation['freq'] = 0
        if self.use_ddp:
            self.batch_accumulation['on'] = False
        if self.opt.use_dp or self.opt.use_ddp:
            self.batch_accumulation['on'] = False

        self.gradient_norm_modisc = []
        self.last_hooks = {}
        if opt.activations_freq is not None:
            self.registerHooks()
        if opt.topK_discrim:
            self.topK_discrim = True
        else:
            self.topK_discrim = False


    def run_generator_one_step(self, data, with_gradients = False):

        self.pix2pix_model_on_one_gpu.switch_disc_grad(False)
        self.pix2pix_model_on_one_gpu.switch_gen_grad(True)

        input_label, input_image, input_style, modalities = util.preprocess_input(data,
                                                                                  self.pix2pix_model_on_one_gpu.name_mod2num,
                                                                                  device = self.device,
                                                                                  label_nc=self.opt.label_nc,
                                                                                  contain_dontcare_label=self.opt.contain_dontcare_label)

        g_losses, generated, accs, g_losses_nw, z = self.pix2pix_model(input_label, input_image, input_style,
                                                                       mode='generator', get_code = True,
                                                                       modalities = modalities)

        self.generated = generated

        if with_gradients:
            # If with gradients is active, we calculate the gradients of the losses specified in
            # pairs with regards to the last layers of the modality discriminator, Generator and
            # discriminator.
            pairs = {'decoder': ['KLD', 'GAN', 'GAN_Feat', 'perceptual', 'self_supervised', 'slice-con'],
                     'encoder': ['KLD', 'GAN', 'GAN_Feat', 'perceptual', 'self_supervised', 'slice-con'],
                     'modisc': ['mod_disc', 'dat_disc']}

            layers = {'decoder': [list(self.pix2pix_model_on_one_gpu.netG.parameters())[-2]],
                      'encoder': [list(self.pix2pix_model_on_one_gpu.netE.parameters())[-2],
                                  list(self.pix2pix_model_on_one_gpu.netE.parameters())[-4]
                                  ]
                      }

            gradients = {}
            for name_loss, val in g_losses.items():
                val.backward(retain_graph = True) # Backward the specific loss
                for structure, layer_params in layers.items():
                    if name_loss in pairs[structure]:
                        try:
                            gradients[
                                "%s_%s" %(name_loss, structure)] = torch.autograd.grad(
                                val, layer_params, retain_graph=True,create_graph=True, allow_unused=True)
                        except:
                            pass
        else:
            g_loss = sum(g_losses.values()).mean()
            g_loss.backward()

        if self.batch_accumulation['on']:
            # Batch Accumulation case
            if self.batch_accumulation['curr_G'] == self.batch_accumulation['freq']:
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()
                self.batch_accumulation['curr_G'] = 0
                torch.cuda.empty_cache()
            else:
                self.batch_accumulation['curr_G'] += 1
        else:
            # Not batch accumulation case
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            torch.cuda.empty_cache()

        self.optimizer_G.step()
        self.g_losses = g_losses
        self.g_losses_noweight = g_losses_nw
        self.d_accuracy = accs

        if with_gradients:
            return gradients

    def run_encoder_one_step(self, data):

        # Freeze the decoder weights
        # Runs a forward pass in encode mode
        # All losses and update

        input_label, input_image, input_style, modalities = util.preprocess_input(data,
                                                                                  self.pix2pix_model_on_one_gpu.name_mod2num,
                                                                                  device = self.device,
                                                                                  label_nc=self.opt.label_nc,
                                                                                  contain_dontcare_label=self.opt.contain_dontcare_label)

        g_losses, generated, accs, g_losses_nw, z = self.pix2pix_model(input_label, input_image, input_style,
                                                                       mode='generator', get_code = True,
                                                                       modalities = modalities)

        self.generated = generated # This needs to go before NMI Loss!!!

        if self.nulldec:
            g_loss = g_losses['KLD']
            if self.pix2pix_model_on_one_gpu.self_supervised:
                g_loss += g_losses['self_supervised']
        else:
            g_loss = sum(g_losses.values()).mean()
        self.d_accuracy = accs
        g_loss.backward()
        if self.batch_accumulation['on']:
            # Batch Accumulation case
            if self.batch_accumulation['curr_G'] == self.batch_accumulation['freq']:
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()
                self.batch_accumulation['curr_G'] = 0
                torch.cuda.empty_cache()
            else:
                self.batch_accumulation['curr_G'] += 1
        else:
            # Not batch accumulation case
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            torch.cuda.empty_cache()

        self.g_losses = g_losses

    def run_discriminator_one_step(self, data, with_gradients = False,
                                   return_predictions = False):

        self.pix2pix_model_on_one_gpu.switch_disc_grad(True)
        self.pix2pix_model_on_one_gpu.switch_gen_grad(False)

        input_label, input_image, input_style, modalities = util.preprocess_input(data,
                                                                                  self.pix2pix_model_on_one_gpu.name_mod2num,
                                                                                  device=self.device,
                                                                                  label_nc=self.opt.label_nc,
                                                                                  contain_dontcare_label=self.opt.contain_dontcare_label)

        if self.topK_discrim:
            if return_predictions:
                d_losses, d_acc, outputs_D = self.pix2pix_model(input_label, input_image, input_style,
                                                                mode='discriminator',
                                                                return_D_predictions = return_predictions,
                                                                modalities = modalities)
                for key, val in outputs_D.items():
                    if type(val) is list:
                        outputs_D[key] = [i.detach().cpu() for i in val]
                    else:
                        outputs_D[key] = val.detach().cpu()
            else:
                d_losses, d_acc = self.pix2pix_model(input_label, input_image, input_style,
                                                     mode='discriminator',
                                                     return_D_predictions = return_predictions,
                                                     modalities = modalities)
        else:
            if return_predictions:
                d_losses, d_acc, outputs_D = self.pix2pix_model(input_label, input_image, input_style,
                                                                mode='discriminator',
                                                                return_D_predictions = return_predictions,
                                                                modalities = modalities)
                for key, val in outputs_D.items():
                    if type(val) is list:
                        outputs_D[key] = [i.detach().cpu() for i in val]
                    else:
                        outputs_D[key] = val.detach().cpu()
            else:
                d_losses, d_acc = self.pix2pix_model(input_label, input_image, input_style,
                                                     mode='discriminator',
                                                     return_D_predictions = return_predictions,
                                                     modalities = modalities)

        # If with gradients is active, we calculate the gradients of the losses specified in
        # pairs with regards to the last layers of the discriminator.

        if with_gradients:

            pairs = {}
            layers = {}
            D_params = list(self.pix2pix_model_on_one_gpu.netD.parameters())
            n_params_per_D = int(len(D_params)/self.opt.num_D)
            for d in range(self.opt.num_D):
                pairs['discriminator_%d' %d] =  ['D_Fake', 'D_real']
                layers['discriminator_%d' %d] = [D_params[n_params_per_D * (d+1) - 2]]

            gradients = {}

            for name_loss, val in d_losses.items():
                val.backward(retain_graph=True)  # Backward the specific loss
                for structure, layer_params in layers.items():
                    if name_loss in pairs[structure]:
                        try:
                            gradients[
                                "%s_%s" % (name_loss, structure)] = torch.autograd.grad(
                                val, layer_params, retain_graph=True, create_graph=True, allow_unused=True)
                        except:
                            pass
        else:
            d_loss = sum(d_losses.values()).mean()
            d_loss.backward()

        if self.batch_accumulation['on']:
            # Batch Accumulation case
            if self.batch_accumulation['curr_D'] == self.batch_accumulation['freq']:
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()
                self.batch_accumulation['curr_D'] = 0
                torch.cuda.empty_cache()
            else:
                self.batch_accumulation['curr_D'] += 1
        else:
            # Not batch accumulation case
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
            torch.cuda.empty_cache()

        self.d_losses = d_losses
        self.d_accuracy = d_acc

        if with_gradients:
            if return_predictions:
                return gradients, outputs_D
            else:
                return gradients
        else:
            if return_predictions:
                return outputs_D

    def record_epoch(self, epoch):
        self.pix2pix_model_on_one_gpu.epoch = epoch

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_losses_nw(self):
        '''
        Get latest losses (not multiplied by any weight factors)
        :return:
        '''

        return {**self.g_losses_noweight, **self.d_losses}

    def get_disc_accuracy(self):
        return self.d_accuracy

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch, device):
        self.pix2pix_model_on_one_gpu.save(epoch, device)
        self.saveOptimizers(epoch)

    def run_tester_one_step(self, data, get_code = False, epoch = None):
        """
        Run one test step on both generators
        :param data:
        :param seqmode: The code for the sequence mode, to choose which generator to test.
        :return:
        """

        with torch.no_grad():
            self.pix2pix_model.eval()
            self.pix2pix_model_on_one_gpu.eval()
            input_label, input_image, input_style, modalities = util.preprocess_input(data,
                                                                                      self.pix2pix_model_on_one_gpu.name_mod2num,
                                                                                      device=self.device,
                                                                                      label_nc=self.opt.label_nc,
                                                                                      contain_dontcare_label=self.opt.contain_dontcare_label)

            validation_losses = {}

            if get_code:
                g_losses, generated, accs, g_losses_nw = self.pix2pix_model(input_label,
                                                                            input_image,
                                                                            input_style,
                                                                            modalities,
                                                                            'validation',
                                                                            )
            else:
                g_losses, generated, accs, g_losses_nw, z = self.pix2pix_model(input_label,
                                                                               input_image,
                                                                               input_style,
                                                                               modalities,
                                                                               'validation',
                                                                               get_code = True,
                                                                               epoch = epoch,)

        self.pix2pix_model.train()
        self.pix2pix_model_on_one_gpu.train()

        if get_code:
            return generated, g_losses, g_losses_nw, z
        else:
            return generated, g_losses, g_losses_nw


    def run_encoder_tester(self, data):
        '''
        Runs the encoder only and returns the codes
        :param data:
        :return:
        '''
        input_label, input_image, input_style, modalities = util.preprocess_input(data,
                                                                                  self.pix2pix_model_on_one_gpu.name_mod2num,
                                                                                  device=self.device,
                                                                                  label_nc=self.opt.label_nc,
                                                                                  contain_dontcare_label=self.opt.contain_dontcare_label)

        if self.opt.type_prior == 'N':
            gen_z, gen_mu, gen_logvar, noise = self.pix2pix_model(input_label,input_image, input_style,
                                                                  modalities,
                                                                  'encode_only_all',)
            return gen_z, gen_mu, gen_logvar, noise
        elif self.opt.type_prior == 'S':
            gen_z, gen_mu, gen_logvar  = self.pix2pix_model(input_label,input_image, input_style,
                                                            modalities,
                                                            'encode_only_all',)
            return gen_z, gen_mu, gen_logvar

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / self.opt.TTUR_factor
                new_lr_D = new_lr * self.opt.TTUR_factor

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def calculate_adaptive_weight(self, recons_loss, gener_loss, last_layer=None):
        if last_layer is not None:
            rec_grass = torch.autograd.grad(recons_loss, last_layer, retain_graph=True)[0]
            gen_grads = torch.autograd.grad(gener_loss, last_layer, retain_graph=True)[0]
        else:
            rec_grass = torch.autograd.grad(recons_loss, self.last_layer[0], retain_graph=True)[0]
            gen_grads = torch.autograd.grad(gener_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(rec_grass) / (torch.norm(gen_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight
        return d_weight

    def saveOptimizers(self, epoch):
        util.save_optimizer(self.optimizer_G, 'G', epoch, self.opt)
        util.save_optimizer(self.optimizer_D, 'D', epoch, self.opt)

    def loadOptimizers(self):
        util.load_optimizer(self.optimizer_G, 'G', self.opt.which_epoch, self.opt)
        util.save_optimizer(self.optimizer_D, 'D', self.opt.which_epoch, self.opt)

    def get_activation(self, name):

         def hook(model, input, output):
             self.last_hooks[name] = output.detach()

         return hook



    def registerHooks(self):


        for D in range(self.opt.num_D):
            if D == 0:
                self.pix2pix_model_on_one_gpu.netD.discriminator_0.model3.register_forward_hook(self.get_activation('disc_0'))
            elif D==1:
                self.pix2pix_model_on_one_gpu.netD.discriminator_1.model3.register_forward_hook(self.get_activation('disc_1'))
            elif D==2:
                self.pix2pix_model_on_one_gpu.netD.discriminator_2.model3.register_forward_hook(self.get_activation('disc_2'))
            elif D==3:
                self.pix2pix_model_on_one_gpu.netD.discriminator_3.model3.register_forward_hook(self.get_activation('disc_3'))
            elif D==4:
                self.pix2pix_model_on_one_gpu.netD.discriminator_4.model3.register_forward_hook(self.get_activation('disc_4'))
            elif D>5:
                ValueError("More than 5 discriminators IS NOT supported by register hooks!")

        self.pix2pix_model_on_one_gpu.netE.fc_mu.register_forward_hook(self.get_activation('enc_mu'))
        self.pix2pix_model_on_one_gpu.netE.fc_mu.register_forward_hook(self.get_activation('enc_sigma'))
        self.pix2pix_model_on_one_gpu.netG.conv_img.register_forward_hook(self.get_activation('decoder'))


