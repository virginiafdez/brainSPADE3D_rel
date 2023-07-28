"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions
import pickle

class TestOptions(BaseOptions):

    def initialize(self, parser):

        BaseOptions.initialize(self, parser)

        parser.add_argument('--root_results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.set_defaults(no_flip=True)
        parser.set_defaults(mode='test')
        parser.set_defaults(isTrain=False)

        # Which tests to run
        parser.add_argument("--test_set_and_ssim", action = 'store_true', help = "Run inference on the test set,"
                                                                                 "including SSIM computation.")
        parser.add_argument("--cross_styles", action = 'store_true', help = "Run cross-styles test on the test set,")
        parser.add_argument("--unseen_modality", action = "store_true", help = "Run inference on unseen modalities."
                                                                               "Plot TSNE representation of "
                                                                               "style encodings")
        parser.add_argument("--cross_modalities", action = "store_true", help = "Latent space plots")

        opt, _ = parser.parse_known_args()

        if opt.test_set_and_ssim:
            parser.add_argument("--max_images", type=int, default=100, help="Maximum number of images (labels) for the "
                                                                            "test_set_and_ssim test")
            parser.add_argument("--regeneration_index", type=int, default=15,
                                help ="Number of times same batch is generated"
                                      "To estimate the variability within the same style.")
        if opt.cross_styles:
            parser.add_argument("--style_dicts_cross_styles", type=str, help="List of styles for the "
                                                                             "cross_styles test.",
                                nargs = '*')
            parser.add_argument("--style_names", type=str, help="For each of the dictionaries / styles,"
                                                                "the name of them.",
                                nargs = '*')

        if opt.cross_modalities:
            parser.add_argument("--unseen_modalities_dict", type=str, help="List of styles belonging to unseen"
                                                                           "modalities (at least a column must"
                                                                           "contain one of those.")

        self.isTrain = False
        return parser

class TuringTestOptions():
    def help(self):
        max_char_name =  30
        max_char_value = 300
        message = ''
        message += 'TestOptions%s\n' %("-"*(max_char_name + max_char_value - 10))
        options = {'image_dir_te': "type: Synthetic or target image directory.",
                   'image_dir_tr': "type: Real or source image directory.",
                   'label_dir_tr': "type: Real or source label directory.",
                   'label_dir_te': "type: Synthetic or target label directory.",
                   'volume': "Whether images are volumetric or not.",
                   'sequences': "List of modalities.",
                   'datasets': 'List of datasets',
                   'max_dataset_size': "Maximum dataset size.",
                   'n_images': "How many images to show for each modality",
                   'use_augmentation': "Whether to use augmentation",
                   'batchSize': "Number of images per batch",
                   'num_workers': "Number of workers",
                   'results_file': "Path to the results file",
                   'skullstrip': "Whether to skullstrip the image"
                   }

        sorted_options = dict(sorted(options.items(), key=lambda x: x[0].lower()))
        for key, value in sorted_options.items():
             message += "|%s%s\t%s%s|\n" %(key, "-"*(max_char_name-len(key)), value, "-"*(max_char_value-len(value)))

        message += "-"*(max_char_name + max_char_value + 3)
        print(message)

    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--image_dir_te', type=str,  help='Synthetic or target image directory')
        parser.add_argument('--image_dir_tr', type=str, help='Real or source image directory.')
        parser.add_argument('--label_dir_tr', type=str, help='Real or source label directory.')
        parser.add_argument('--label_dir_te', type=str, help='Synthetic or target label directory.')
        parser.add_argument('--volume', action = 'store_true', help = "Whether images are volumetric or not")
        parser.add_argument('--sequences', type = str, nargs='*', help = "List of modalities")
        parser.add_argument('--datasets', type=str, nargs='*', help="List of datasets")
        parser.add_argument('--max_dataset_size', type=int, default=10000,  help='Maximum dataset size.')
        parser.add_argument('--n_images', type = int, default = 300, help = "How many images to show for each modality")
        parser.add_argument('--use_augmentation', action = 'store_true', help = "Whether to use augmentation")
        parser.add_argument('--batchSize', type = int, default = 8, help = "Number of images per batch")
        parser.add_argument('--num_workers', type = int, default = 0, help = "Number of workers")
        parser.add_argument('--results_dir', type=str, help = "Results directory")
        parser.add_argument('--skullstrip', action = 'store_true', help = "Whether to skull strip image")

        self.isTrain = False
        return parser

class FullPipelineOptions(BaseOptions):

    def load_options(self, file, dataset_type = 'sliced'):

        new_opt = pickle.load(open(file + '.pkl', 'rb'))

        #Remove training parameters
        new_opt.isTrain = False
        new_opt.semantic_nc = new_opt.label_nc + \
            (1 if new_opt.contain_dontcare_label else 0) + \
            (0)
        # set gpu ids
        str_ids = new_opt.gpu_ids.split(',')
        new_opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                new_opt.gpu_ids.append(id)

        assert len(new_opt.gpu_ids) == 0 or new_opt.batchSize % len(new_opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (new_opt.batchSize, len(new_opt.gpu_ids))

        if new_opt.phase == 'train' and new_opt.batchSize < 4:
            new_opt.latent_triplet_loss = False
            print("Triplet loss deactivated because batch size is insufficient.")


        if dataset_type == 'volume' and new_opt.dataset_type == 'sliced':
            # Volumetric dataset - specific parameters.
            new_opt.dataset_type = 'volume'
            new_opt.intensify_lesions = 0
            new_opt.store_and_use_slices = True
            new_opt.lesion_sampling_mode = 'threshold'
            new_opt.threshold_low = 100
            new_opt.ruleout_offset = 0.25
            new_opt.sample_lesions = False
            new_opt.continuous_slices = True

        return new_opt

