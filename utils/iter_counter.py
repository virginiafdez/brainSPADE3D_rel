"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import time
import numpy as np


# Helper class that keeps track of training iterations
class IterationCounter():
    def __init__(self, opt, dataset_size):

        self.opt = opt
        self.dataset_size = dataset_size
        self.first_epoch = 1
        self.total_epochs = opt.niter + opt.niter_decay
        self.epoch_iter = 0  # iter number within each epoch
        self.iter_record_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'iter.txt')
        self.stored_accuracy = [0]*opt.steps_accuracy
        self.max_length = opt.steps_accuracy
        self.stored_losses = {}
        self.gradient_step_gen = 0
        self.gradient_step_dis = 0

        if opt.isTrain and opt.continue_train:
            try:
                self.first_epoch, self.epoch_iter = np.loadtxt(
                    self.iter_record_path, delimiter=',', dtype=int)
                print('Resuming from epoch %d at iteration %d' % (self.first_epoch, self.epoch_iter))
            except:
                print('Could not load iteration record at %s. Starting from beginning.' %
                      self.iter_record_path)

        self.total_steps_so_far = (self.first_epoch - 1) * dataset_size + self.epoch_iter

    # return the iterator of epochs for the training
    def training_epochs(self):
        return range(self.first_epoch, self.total_epochs + 1)

    def store_losses(self, g_losses = None, d_losses = None, accuracies = None):
        if g_losses is not None:
            for loss, value in g_losses.items():
                if loss in self.stored_losses.keys():
                    if len(self.stored_losses[loss]) == self.max_length:
                        self.stored_losses[loss] = self.stored_losses[loss][1:] + [value.mean().item()]
                    else:
                        self.stored_losses[loss].append(value.mean().item())
                else:
                    self.stored_losses[loss] = [value.mean().item()]
        if d_losses is not None:
            for loss, value in d_losses.items():
                if loss in self.stored_losses.keys():
                    if len(self.stored_losses[loss]) == self.max_length:
                        self.stored_losses[loss] = self.stored_losses[loss][1:] + [value.mean().item()]
                    else:
                        self.stored_losses[loss].append(value.mean().item())
                else:
                    self.stored_losses[loss] = [value.mean().item()]
        if accuracies is not None:
            for loss, value in accuracies.items():
                if loss in self.stored_losses.keys():
                    if len(self.stored_losses[loss]) == self.max_length:
                        self.stored_losses[loss] = self.stored_losses[loss][1:] + [value.mean().item()]
                    else:
                        self.stored_losses[loss].append(value.mean().item())
                else:
                    self.stored_losses[loss] = [value.mean().item()]

    def getStoredLosses(self):
        out = {}
        for key, value in self.stored_losses.items():
            out[key] = np.mean(value)
        #out['D_acc_total'] = np.mean(self.stored_accuracy)
        return out

    def add_assess_accuracy(self, accuracy):
        if accuracy is None:
            accuracy = 0.0
        self.stored_accuracy = self.stored_accuracy[1:] + [accuracy]
        return np.mean(self.stored_accuracy)

    def record_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        self.epoch_iter = 0
        self.last_iter_time = time.time()
        self.current_epoch = epoch

    def record_one_iteration(self):
        current_time = time.time()

        # the last remaining batch is dropped (see data/__init__.py),
        # so we can assume batch size is always opt.batchSize
        self.time_per_iter = (current_time - self.last_iter_time) / self.opt.batchSize
        self.last_iter_time = current_time
        self.total_steps_so_far += self.opt.batchSize
        self.epoch_iter += self.opt.batchSize

    def record_one_gradient_iteration_gen(self):
        self.gradient_step_gen += self.opt.batchSize

    def record_one_gradient_iteration_dis(self):
        self.gradient_step_dis += self.opt.batchSize

    def record_epoch_end(self):
        current_time = time.time()
        self.time_per_epoch = current_time - self.epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (self.current_epoch, self.total_epochs, self.time_per_epoch))
        if self.current_epoch % self.opt.save_epoch_freq == 0:
            np.savetxt(self.iter_record_path, (self.current_epoch + 1, 0),
                       delimiter=',', fmt='%d')
            print('Saved current iteration count at %s.' % self.iter_record_path)

    def record_current_iter(self):
        np.savetxt(self.iter_record_path, (self.current_epoch, self.epoch_iter),
                   delimiter=',', fmt='%d')
        print('Saved current iteration count at %s.' % self.iter_record_path)

    def needs_saving(self):
        return ((self.total_steps_so_far / self.opt.batchSize) % self.opt.save_latest_freq) == 0

    def needs_gradient_calc(self, for_disc = False):
        if self.opt.gradients_freq is None:
            return False
        else:
            if for_disc:
                return (np.round(self.gradient_step_dis / self.opt.batchSize) % self.opt.gradients_freq) == 0
            else:
                return (np.round(self.gradient_step_gen / self.opt.batchSize) % self.opt.gradients_freq) == 0

    def needs_D_display(self):
        return (np.round(self.gradient_step_dis / self.opt.batchSize) % self.opt.disp_D_freq) == 0

    def needs_activations(self, for_disc = False):
        if self.opt.gradients_freq is None:
            return False
        else:
            if for_disc:
                return (np.round(self.gradient_step_dis / self.opt.batchSize) % self.opt.activations_freq) == 0
            else:
                return (np.round(self.gradient_step_gen / self.opt.batchSize) % self.opt.activations_freq) == 0

    def needs_printing(self):
        return ((self.epoch_iter / self.opt.batchSize) % self.opt.print_freq) == 0
        #return (self.epoch_iter / self.opt.batchSize % self.opt.print_freq) == 0
        #return (self.total_steps_so_far % self.opt.print_freq) < self.opt.batchSize

    def needs_enc_display(self):
        return (self.total_steps_so_far / self.opt.batchSize %self.opt.display_enc_freq) == 0
        #return (self.epoch_iter / self.opt.batchSize % self.opt.display_enc_freq) == 0

    def needs_displaying(self):
        #return (self.epoch_iter/self.opt.batchSize % self.opt.display_freq) == 0
        return (self.epoch_iter/self.opt.batchSize) % self.opt.display_freq == 0
        #return (self.total_steps_so_far % self.opt.display_freq) < self.opt.batchSize

    def needs_testing(self):
        return (self.current_epoch % self.opt.test_freq) < self.opt.batchSize
