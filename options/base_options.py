"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os
from utils import util
import torch
import models_spade
import data
import pickle
import shutil

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def help(self):
        max_char_name =  30
        max_char_value = 300

        message = ''
        title = "BASE OPTIONS"
        message += '%s%s\n' %(title, "-"*(max_char_name + max_char_value - 3 - len(title)))
        options = {
            'DATA': {
                'data_dict' : 'type: string, help: path to the TSV file that contains labels and images',
                'style_dict': 'type: str, help: path to the TSV file that contains style labels and images',
                'skullstrip': 'type: flag, help: Whether to skull-strip or not the images',
                'cache_dir': 'type: string, default: None, help: Directory where you can cache images and label volumes with PersistentDataset',
                'cache_type': 'type: string, default: none, help:  '
                              'PeristentDataset is used (caching to disk),if none, Dataset is used (no caching)',
                'non_corresponding_dirs': 'type: flag, help: Whether the label directory is different from the image directory.'
                                          'Since the files are not equivalent, non_corresponding_style is True.'
                                          'You need to provide style_label_dir for skullstrip mode',
                'non_corresponding_ims': 'type: flag, Whether to use different style images within the same label and image dirs.',
                'sequences': 'type: list of strings, default: [T1, FLAIR], help: List of sequences that make up the dataset. Must coincide with the label.',
                'fix_seq': 'type: str, default: None, help: List of sequences that make up the dataset. Must coincide with the label.',
                'max_dataset_size': 'type: int, default: none, help: Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.',
                'crop_size': 'type: list of int, default: [256, 256], help: New height and width to resize the images to.',
                'label_nc': 'type: int, default: 182, help: # of input label classes without unknown class. If you have '
                            'unknown class as class label, specify --contain_dopntcare_label',
                'contain_dontcare_label': 'DEPRECATED type: flag, help: if the label map contains dontcare label (dontcare=255)'
            },

            'MODEL' : {
                'model': 'type: string, default: pix2pix, help: which model to use',
                'norm_G': 'type: string, default: spectralinstance, help: instance normalization or batch normalization',
                'norm_E': 'type: string, default: spectralinstance, help: instance normalization or batch normalization',
                'netG': 'type: string, default: spade, help: selects model to use for netG (pix2pixhd | spade)',
                'ngf': 'type: int, default: 64, help: # of gen filters in first conv layer',
                'init_type' : 'type: string, default: xavier, help: network initialization [normal|xavier|kaiming|orthogonal]',
                'init_variance': 'type: float, default: 0.02, help: variance of the initialization distribution',
                'z_dim': 'type: int, default: 256, help: dimension of the latent z vector',
                'n_decoders': 'type: string, default: dontcare, help: Names of the decoder tails, with dashes in between: i.e. FLAIR-T1',
                'upsampling_type': 'type: Type of convolution type: transposed upsample subpixel',
                'nef': 'type: int, defualt: 16, help: # of encoder filters in the first conv layer',
                'use_vae' : 'type: flag, help: enable training with an image encoder.',
                'type_prior': 'type: string, default: N,  help: Type of prior, S - spherical, uniform, or N - normal',
            },

            'EXPERIMENT': {
                'name': 'type: string, default: brainspade, help: name of the experiment. It decides where to store samples and models_',
                'gpu_ids': 'type: string, default: 0, help: gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU',
                'checkpoints_dir': 'type: string, default: checkpoints, help: models_ are saved here',
                'mode': 'type: string, default: checkpoints, help: train, val, test, etc',
                'batchSize': 'type: int, default: 1, help: input batch size',
                'output_nc': 'type: int, default: 3, help: # of output image channels',
                'serial_batches': 'type: flag, default: checkpoints, help: if true, takes images in order to make batches, otherwise takes them randomly',
                'nThreads': 'type: int, default: 0, help: number of threads for loading data',
                'load_from_opt_file': 'DEPRECATED type: flag, help: load the options from checkpoints and use that as default',
                'cache_filelist_write': 'DEPRECATED type: flag, help:saves the current filelist into a text file, so that it loads faster',
                'cache_filelist_read': 'DEPRECATED type: flag, help: reads from the file list cache',
                'display_winsize': 'type: int, default: 400, help: display window size',

            },
            }

        for SUPERKEY, SUPERVALUE in options.items():
            sub_title = "%s" %SUPERKEY
            sub_message = ""
            sub_message += '%s%s\n' %(sub_title, "-"*(max_char_name + max_char_value - 3 - len(sub_title)))
            sorted_super =  dict(sorted(SUPERVALUE.items(), key=lambda x: x[0].lower()))
            for key, value in sorted_super.items():
                sub_message += "|%s%s\t%s%s|\n" %(key, "-"*(max_char_name-len(key)), value, "-"*(max_char_value-len(value)))
            message += sub_message

        message += "-"*(max_char_name + max_char_value + 3)
        print(message)

    def initialize(self, parser):

        # experiment specifics
        parser.add_argument('--name', type=str, default='brainspade', help='name of the experiment. It decides where to store samples and models_')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models_ are saved here')
        parser.add_argument('--mode', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

        # for model
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--z_dim', type=int, default=256,
                            help="dimension of the latent z vector")
        parser.add_argument('--n_decoders', type = str, default = "dontcare", help = 'Names of the decoder tails, with dashes in between: i.e. FLAIR-T1')
        parser.add_argument('--upsampling_type', type = str, default="upsample", help = "Type of convolution type: transposed upsample subpixel")
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')
        parser.add_argument('--type_prior', default='N', help = 'Type of prior, S - spherical, uniform, or N - normal.')

        # DATASET features
        parser.add_argument('--data_dict', type=str, required=True,
                            help='path to the TSV file that contains labels and images')
        parser.add_argument('--style_dict', type=str, default=None,
                            help='path to the TSV file that contains style labels and images')

        parser.add_argument('--skullstrip', action = 'store_true', help= "Whether to skull-strip or not the images.")
        parser.add_argument('--cut', type=str, default='a', help = "a, c or s (sagittal, coronal or axial) cuts for the 2D"
                                                                   "encoding of styles")

        parser.add_argument('--cache_dir', type =str, default = None, help = "Directory where you can cache images"
                                                                     "and label volumes with PersistentDataset")
        parser.add_argument('--cache_type', type = str, default = "none", help = "if cache, PeristentDataset is used (caching to disk),"
                                                                                 "if none, Dataset is used (no caching)")
        parser.add_argument('--non_corresponding_ims', action = 'store_true', help="Whether to use different style images within "
                                                                                   "the same label and image dirs.")
        parser.add_argument('--non_corresponding_dirs', action='store_true', help="Whether to use different style images"
                                                                                 "than the ones in label_dir and images_dir.")
        parser.add_argument('--sequences', type = str, nargs ='*',  default =["T1", "FLAIR"],
                            help = "List of sequences that make up the dataset. Must coincide with the label. ")
        parser.add_argument('--fix_seq', type = str, default= None, help = "In case you want a fix modality to be picked up, specify the modality. ")
        parser.add_argument('--max_dataset_size', type=int, default=10000000,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--crop_size', type=int, nargs='*', default=[256, 256, 120],
                            help='New height and width to resize the images to.')
        parser.add_argument('--label_nc', type=int, default=182,
                            help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--contain_dontcare_label', action='store_true',
                            help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--style_slice_consistency', action='store_true', help="Style slice consistency.")
        parser.add_argument('--shuffle', action='store_true', help = "Whether to shuffle labels in training."
                                                                     "This won't affect interleaved modality selection.")
        parser.add_argument('--chunk_size', type = int, default = None, help = "If, along the axial dimension, you want to select random"
                                                               "chuncks of [chunk_size] shape.",)
        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models_spade.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)


        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)
        opt = parser.parse_args()
        self.parser = parser

        # Overwrite some options
        opt.use_vae = True
        #opt.sequences = ["T1", "FLAIR"]

        if opt.n_decoders != "dontcare":
            opt.n_decoders = opt.n_decoders.split("-") # We split

        return opt

    def print_options(self, opt):

        if opt.mode == 'test' or not opt.use_ddp:
            message = ''
            message += '----------------- Options ---------------\n'
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            message += '----------------- End -------------------'
            print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        # Save previous options before
        if opt.continue_train and os.path.isfile(file_name+'.txt'):
            # Backup
            old_filename = file_name.replace('opt', 'opt_old')
            shutil.copy(file_name+'.txt', old_filename+'.txt')

        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        #if len(opt.gpu_ids) > 0:
        #    torch.cuda.set_device(opt.gpu_ids[0])

        if opt.mode == 'test' or (opt.use_dp and not opt.use_ddp):
            assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
                "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
                % (opt.batchSize, len(opt.gpu_ids))

        if opt.mode not in ['train', 'test']:
            ValueError("mode must be train or test")
        if opt.mode == 'train' and opt.batchSize < 4:
            opt.latent_triplet_loss = False
            print("Triplet loss deactivated because batch size is insufficient.")
        if opt.mode == 'test':
            opt.augment = False

        self.opt = opt
        return self.opt
