"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
import numpy as np
import moreutils as uvir

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            self.disc_im_dir = os.path.join(self.web_dir, 'D_outputs')
            #print('create web directory %s...' % self.web_dir)
            if not os.path.isdir(self.web_dir):
                os.makedirs(self.web_dir)
            if not os.path.isdir(self.img_dir):
                os.makedirs(self.img_dir)
            if not os.path.isdir(self.disc_im_dir):
                os.makedirs(self.disc_im_dir)
            self.initialize_CheckErrors()
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
            self.drop_first = opt.drop_first

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)
                
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s_%d.png' % (epoch, step, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.3d_%s.png' % (epoch, step, label))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]

                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.3d_%s_%d.png' % (n, step, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.3d_%s.png' % (n, step, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            #v = v.mean().float()
            message += '%s: %.3f; ' % (k, v)
        print(message)

        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def initialize_Validation(self, ct):

        self.val_dir = os.path.join(self.web_dir, 'validation_results')
        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)
        if not ct:
            self.initialize_Results_File()

    def initialize_Results_File(self):
        with open(os.path.join(self.web_dir, 'validation_results', 'results_training.txt'), 'w') as f:
            header_line = "Epoch\tSSIM\tModAcc\tDatAcc\n"
            f.write(header_line)
            f.close()

    def register_Test_Results(self, epoch, results):

        newline = "%d\t" %epoch
        for name, res in results.items():
            newline = newline + "%.3f\t" %res

        with open(os.path.join(self.web_dir, 'validation_results', 'results_training.txt'), 'a') as f:
            f.write(newline+"\n")
            f.close()

    def initialize_CheckErrors(self):
        self.check_errors_dir = os.path.join(self.web_dir, 'errors_brainspade')
        if not os.path.exists(self.check_errors_dir):
            os.makedirs(self.check_errors_dir)

    def save_codes(self, z, mu, logvar, sequence, noise = None , epoch = 0, iter = 0):
        '''
        Saves the code, mu and log variance of each element of a batch in a separate
        image.
        :param z: Code of a specific image (output of encoder)
        :param mu: Mean of the encoder distribution
        :param logvar: Log var of the encoder distribution
        :param noise: Noise vector.
        :param sequence: Sequence of the particular batch (FLAIR or T1)
        :return:
        '''
        self.code_dir = os.path.join(self.web_dir, 'code_plots')
        if not os.path.isdir(os.path.join(self.web_dir, 'code_plots')):
            os.makedirs(self.code_dir)

        z = z.detach().cpu().numpy()
        mu = mu.detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()
        std = np.exp(0.5 * logvar)
        if noise is not None:
            noise = noise.detach().cpu().numpy()

        for b_ind in range(z.shape[0]):
            savename = os.path.join(self.code_dir, "%s_ep%d_it%d_code_%d.png" %(sequence[b_ind], epoch,iter, b_ind))
            if noise is not None:
                out_data = [z[b_ind, ...], mu[b_ind, ...], std[b_ind, ...], noise[b_ind, ...]]
            else:
                out_data = [z[b_ind, ...], mu[b_ind, ...], std[b_ind, ...]]
            uvir.codeStats(out_data, savename)

    def register_Val_Losses(self, epoch, errors_nw = None, errors_w = None, print_it = False):
        '''
        Writes the weighted and unweighted losses in the validation_results directory.
        :param epoch:
        :param errors:
        :param t:
        :return:
        '''

        # Unweighted losses
        if errors_nw is not None:
            if not os.path.isfile(os.path.join(self.web_dir, 'validation_results', 'validation_losses_NW.txt')):
                with open(os.path.join(self.web_dir, 'validation_results', 'validation_losses_NW.txt'), "w") as log_file:
                    now = time.strftime("%c")
                    log_file.write('================ Validation Loss Unweighted (%s) ================\n' % now)
            with open(os.path.join(self.web_dir, 'validation_results', 'validation_losses_NW.txt'), "a") as log_file:
                message = '(epoch: %d, iters: %d) ' % (epoch, 1)
                for k, v in errors_nw.items():
                    message += '%s: %.3f; ' % (k, v)
                message+="\n"
                log_file.write(message)
                log_file.close()
            if print_it:
                print("Validation losses (unweighted): %s" %message)

        # Weighted Losses
        if errors_w is not None:
            if not os.path.isfile(os.path.join(self.web_dir, 'validation_results', 'validation_losses_W.txt')):
                with open(os.path.join(self.web_dir, 'validation_results', 'validation_losses_W.txt'),
                          "w") as log_file:
                    now = time.strftime("%c")
                    log_file.write('================ Validation Loss Unweighted (%s) ================\n' % now)
            with open(os.path.join(self.web_dir, 'validation_results', 'validation_losses_W.txt'),
                      "a") as log_file:
                message = '(epoch: %d, iters: %d) ' % (epoch, 1)
                for k, v in errors_w.items():
                    message += '%s: %.3f; ' % (k, v)
                message += "\n"
                log_file.write(message)
                log_file.close()

    def back_up_validation_slices(self):
        '''
        Check whether there is a back up file in the validation folder.
        :return:
        '''
        if self.opt.dataset_type == 'sliced': # With sliced dataset we do not record the slices (we don't need them!)
            return  True
        else:
            return os.path.isfile(os.path.join(self.val_dir, 'back_up_stored_slices.txt'))

    def plot_D_results(self, outputs_D, epoch, iter = 1):

        '''
        Plots discriminator results.
        '''

        if self.drop_first:
            outputs_D['pred_real'] = outputs_D['pred_real'][1:]
            outputs_D['pred_fake'] = outputs_D['pred_fake'][1:]
        uvir.plot_D_outputs(outputs_D, self.disc_im_dir, "discrim_results_epoch_%d_%d.png" %(epoch, iter),
                            "Results for epoch %d, iter %d" %(epoch, iter))







