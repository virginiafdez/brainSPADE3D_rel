from torch.utils.tensorboard import SummaryWriter
import torch
import os
import numpy as np

class BrainspadeBoard:

    def __init__(self, opt):

         self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'log_dir')
         if not os.path.isdir(self.log_dir):
             os.makedirs(self.log_dir)
         self.writer = SummaryWriter(log_dir=self.log_dir)
         self.losses = {'GAN_LOSSES':['GAN', 'D_real', 'D_Fake', 'GAN_Feat',
                                      'D_acc_reals', 'D_acc_fakes'],
                        'Encoder': ['KLD', 'self_supervised', 'slice-con'],
                        'Other':["perceptual"]}

    def log_results(self, losses, epoch, is_val = False):

        if is_val:
            scope_main = "validation"
        else:
            scope_main = "train"

        for scope, list_losses in self.losses.items():
            for loss in list_losses:
                if loss in losses.keys():
                    loss_value = losses[loss]
                else:
                    loss_value = torch.tensor(np.nan)
                self.writer.add_scalar("%s/%s/%s" %(scope_main, scope, loss), loss_value, epoch)

    def close_board(self):
        self.writer.flush()
        self.writer.close()

    def log_grad_histograms(self, gradients, epoch, iteration, niter_per_epoch):

        # Reorder gradients
        grads = {'decoder': {}, 'encoder': {}, 'discriminator':{}}
        for scope in grads.keys():
            for key, value in gradients.items():
                if scope in key:
                    grads[scope][key] = value

        #
        for scope in grads.keys():
            for loss_item, loss_item_val in grads[scope].items():
                for param in loss_item_val:
                    if param is None:
                        continue
                    if len(param.shape) == 1:
                        name_layer = 'bias'
                    elif len(param.shape) == 4:
                        name_layer  = 'conv'
                    else:
                        name_layer = 'linear'
                    hist_name = '%s/HIST_%s_%s' %(scope,loss_item, name_layer)
                    self.writer.add_histogram(hist_name, param, epoch*niter_per_epoch+iteration)
                    #writer.add_scalar(scalar_name, torch.norm(f.grad.data).item(), e)

    def log_act_histograms(self, activations, epoch, iteration, niter_per_epoch):

        scope = 'activations'

        for activation_name, activation in activations.items():
            #If it's a discrminator activation, we separate reals from fakes.
            if 'disc' in activation_name:
                bs = int(activation.shape[0]/2)
                fakes = activation[:bs, ...]
                reals = activation[bs:, ...]
                hist_name_reals = '%s/activation_%s_reals' %(scope, activation_name)
                self.writer.add_histogram(hist_name_reals, reals, epoch * niter_per_epoch + iteration)
                hist_name_fakes = '%s/activation_%s_fakes' % (scope, activation_name)
                self.writer.add_histogram(hist_name_fakes, fakes, epoch * niter_per_epoch + iteration)
            else:
                hist_name = '%s/activation_%s' %(scope, activation_name)
                self.writer.add_histogram(hist_name, activation, epoch*niter_per_epoch+iteration)

