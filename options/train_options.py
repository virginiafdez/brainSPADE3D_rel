"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def help(self):
        BaseOptions().help()
        max_char_name =  30
        max_char_value = 300
        message = ''
        title = "TRAIN OPTIONS"
        message += '%s%s\n' %(title, "-"*(max_char_name + max_char_value - 3 - len(title)))
        options = {
            'DATA': {
              'perc_val': 'type: float, default: 0.15, help: Percentage of training set left for validation',
              'augment': 'type: flag, default = True, help: Do augmentation during training. ',
              'data_dict_val': 'Path to the TSV file that contains validation labels and images.'
                               'If None, it will be calculated out of the data_dict.',
              'style_dict_val': 'Path to the TSV file that contains validation style labels and images.'
                                'If None, it will be derived from the original style_dict.'

            },
            'DISPLAYS AND TRAINING PROCESS': {
                'display_freq': 'type: int, default: 100, help: frequency of showing training results on screen',
                'test_freq': 'type: int, default: 1, help: frequency of testing resuts (in epochs)',
                'print_freq': 'type: int, default: 100, help: frequency of showing training results on console',
                'save_latest_freq': 'type: int, default: 5000, help: frequency of saving the latest results',
                'save_epoch_freq': 'type: int, default: 10, help: frequency of saving checkpoints at the end of epochs',
                'save_epoch_copy': 'type: int, default: 10, help: ffrequency of saving checkpoints at the end of epochs',
                'no_html': 'type: flag, help: do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/',
                'debug': 'type: flag, help: only do one epoch and displays at each iteration',
                'tf_log': 'type: flag, help: if specified, use tensorboard logging. Requires tensorflow installed',
                'print_grad_freq': 'type: int, default: 5000, help: Gradient norm of the modality discrimination layers.',
                'disp_D_freq': 'type: int, default: 5000, help: Gradient norm of the modality discrimination layers.',
                'gradients_freq': 'type: int, default: None, help: Frequency of calculation of gradients of the losses wr to last layers of '
                                  'the network. If None, they are not calculated at all.',
                'activations_freq': 'type: int, default: None, help: Frequency of calculation of gradients of the losses wr to last layers of '
                                    'the network. If None, they are not calculated at all.',
                'use_tboard': 'type: flag, help: Use tensorboard Summary Writer. Warning: can te a lot of memory!',
                'tboard_gradients': 'type: flag, help: Plot gradients on tensorboard',
                'tboard_activations': 'type: flag, help: Plot activations on tensorboard',
                'display_enc_freq': 'type: int, default:, 100, help: Frequency (in iterations) at which we save codes.'
            },

            'TRAINING LOSSES AND PROCESS' : {
                'niter': 'type: int, default: 50, help: # of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay',
                'niter_decay': 'type: int, default: 0, help: # of iter to linearly decay learning rate to zero',
                'optimizer': 'type: str, default: adam',
                'beta1': 'type: float, default: 0.0, help: momentum of term adam',
                'beta2': 'type: float, default: 0.9, help: momentum of term adam',
                'no_TTUR': 'type: flag, help: use TTUR training scheme',
                'TTUR_factor': 'type: int, default: 2, help: factor of downgrading LR of generator wr to discriminator',
                'topK_discrim': 'type: flag, help: use a top-K filtering to train adversarial set up (Sinha et al.)',
                'lambda_feat': 'type: float, default: 10.0, help: weight for feature matching loss',
                'lambda_perceptual': 'type: float, default: 10.0, help: weight for the vgg loss',
                'lambda_slice_consistency': 'type: float, default: 0.0, help: Weight given to slice by slice consistency',
                'activation_slice_consistency': 'type: int, default: 100, help: Epoch in which you want slice consistency to get activated',
                'no_ganFeat_loss': 'type: flag, help: if specified, do *not* use discriminator feature matching loss',
                'no_vgg_loss': 'type: flag, help: if specified, do *not* use VGG feature matching loss',
                'gan_mode': 'type:str, default: hinge, help: (ls|original|hinge)',
                'lambda_kld': 'type: float, default: 0.05, help: weight given to KLD loss',
                'train_enc_only': 'type: int, default: None, help: From this epoch onwards, train only the encoder (and not the decoder)',
                'batch_acc_fr': 'type: int, default: 0, help: Frequency of batch accumulation.',
                'mod_disc_dir': 'type: string, default: none, help: Path to modality discrminator directory. If mod_disc_path is specified, disregarded.',
                'disc_acc_lowerth': 'type: float, default: 0.65, help: Accuracy (0-1) of the discriminator below which discriminator only is trained.',
                'disc_acc_upperth': 'type: float, default: 0.85, help: Accuracy (0-1) of the discriminator above which generator only is trained.',
                'steps_accuracy': 'type: int, default: 20, help: Number of iterations on which to base the accuracy calculation.',
                'pretrained_E': 'type: string, default: None, help: Path to pretrained encoder if needed',
                'freezeE': 'type: boolean, default: false, help: freeze weights of the encoder',
                'self_supervised_training': 'type: float, default: 0.0, help: Weight given to self-supervised loss.',
                'distance_metric': 'type: string, default: l1, help: Distance metric used for the self-supervised loss.',
                'lr': 'type: float, default: 0.0002, help: initial learning rate for adam',
                'D_steps_per_G': 'type: int, default: 1, help: number of discriminator iterations per generator iterations.',
                'continue_train': 'type: flag, help: continue training: load the latest model.',
                'which_epoch': 'type: string, default: latest, help: which epoch to load? set to latest to use latest cached model.',
                },

            'DISCRIMINATOR': {
                'norm_D': 'type: string, default: spectralinstance, help: instance normalization or batch normalization',
                'drop_first': 'type: flag, help: Drop first discriminator (smaller receptive field) when calculating the loss.',
                'ndf': 'type: int, default: 64, help: # of discrim filters in first conv layer',
                'netD': 'type: string, default: multiscale, help: (n_layers|multiscale|image)',
                'D_modality_class': 'type: flag, help: Flag that incorporates the modality to the discriminator input.'
            }

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
        BaseOptions.initialize(self, parser)
        # Training process
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--test_freq', type=int, default=1, help = 'frequency of testing resuts (in epochs)')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_epoch_copy', type=int, default=10,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        parser.add_argument('--print_grad_freq', type = int, default = 5000, help = 'Gradient norm of the modality discrimination'
                                                                                    'layers.')
        parser.add_argument('--disp_D_freq', type = int, default = 600, help = "Frequency of display of discriminator outputs.")
        parser.add_argument('--gradients_freq', type = int, default = None,
                            help = "Frequency of calculation of gradients of the losses wr to last layers of"
                                   "the network. If None, they are not calculated at all.")
        parser.add_argument('--activations_freq', type = int, default = None,
                            help = "Frequency of calculation of gradients of the losses wr to last layers of"
                                   "the network. If None, they are not calculated at all.")
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--use_tboard', action = 'store_true', help = "Use tensorboard Summary Writer. Warning: can te a lot of memory!")
        parser.add_argument('--tboard_gradients', action='store_true',
                            help="Plot gradients on tensorboard")
        parser.add_argument('--tboard_activations', action='store_true',
                            help="Plot activations on tensorboard")
        parser.add_argument('--display_enc_freq', type=int, default = 100, help="Frequency (in iterations) at which "
                                                                           "we save codes.")
        parser.add_argument('--use_ddp', action = 'store_true',
                            help = "Use distributed data parallel")
        parser.add_argument('--use_dp', action='store_true',
                            help="Use distributed data parallel")

        # Loss functions and optimiser
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--TTUR_factor', type = int, default = 2,
                            help = "Factor of downgrading LR of generator wr to discriminator.")
        parser.add_argument('--topK_discrim', action='store_true', help = 'Use a Top-K filtering to train the discriminator'
                                                                          'and the generator.')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_perceptual', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--lambda_slice_consistency', type=float, default=0.0, help='Weight given to slice'
                                                                                        'by slice consistency.')
        parser.add_argument('--activation_slice_consistency', type=int, default=100, help="epoch in which you "
                                                                                          "want slice consistency"
                                                                                          "to get activated")
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator '
                                                                           'feature matching loss')
        parser.add_argument('--no_perceptual_loss', action='store_true',
                            help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--lambda_kld', type=float, default=0.05)
        parser.add_argument('--lambda_style_slice_consistency', type=float, default=1.0, help="Weight for slice consistency")
        parser.add_argument('--train_enc_only', type =int, default = None, help="From this epoch onwards, train only"
                                                                                 "the encoder (and not the decoder)")
        parser.add_argument('--batch_acc_fr', type=int, default = 0, help="Frequency of batch accumulation.")
        parser.add_argument('--mod_disc_dir', type = str, default=None, help = "Optional path specified for the modality discriminator."
                                                                               "Otherwise, checkpoints is picked. ")
        parser.add_argument('--disc_acc_lowerth', type = float, default = 0.65, help = "Accuracy (0-1) of the discriminator"
                                                                                       "below which discriminator only is trained.")
        parser.add_argument('--disc_acc_upperth', type = float, default = 0.85, help = "Accuracy (0-1) of the discriminator"
                                                                                       "above which generator only is trained.")
        parser.add_argument('--steps_accuracy', type = int, default = 20, help="Number of iterations on which to base the"
                                                                               "accuracy calculation. ")
        parser.add_argument('--pretrained_E', type=str, default=None, help="Path to pretrained encoder if needed.")
        parser.add_argument('--freezeE', type = bool, default = False, help = "Freeze weights of the encoder.")
        parser.add_argument('--self_supervised_training', type=float, default=0.0,
                            help="Weight given to self-supervised loss. ")
        parser.add_argument('--distance_metric', type=str, default='l1',
                            help="Distance metric used for the self-supervised loss ")
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--perceptual_loss_path', type=str, default=None, help = "Path to the perceptual loss weights"
                                                                                     "if preloaded = True")

        # for discriminators
        parser.add_argument('--norm_D', type=str, default='spectralinstance',
                            help='instance normalization or batch normalization')
        parser.add_argument('--drop_first', action = 'store_true', help = "Drop first discriminator (smaller receptive field)"
                                                                          "when calculating the loss. ")
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')

        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--D_modality_class', action='store_true', help = "Flag that incorporates the modality to the "
                                                                              "discriminator input. ")
        # for data
        parser.add_argument('--perc_val', type = float, default = 0.15,
                            help = "Percentage of training set left for validation")
        parser.add_argument('--augment', action = 'store_true', help = "Whether to perform augmentation during validation. ")
        parser.add_argument('--data_dict_val', type=str, default = None,
                            help='path to the TSV file that contains validation labels and images.'
                                 'If None, it will be calculated out of the data_dict.')
        parser.add_argument('--style_dict_val', type=str, default=None,
                            help='path to the TSV file that contains validation style labels and images.'
                                 'If None, it will be derived from the original style_dict.')


        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        if opt.use_ddp:
            opt.use_dp = False

        self.isTrain = True
        return parser
