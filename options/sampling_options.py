"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions

class SamplingOptions(BaseOptions):
    def initialize(self, parser):
        parser.add_argument('--model_path', type=str, help='Absolute path leading to the model (pth file) and option txt or pkl file.')
        parser.add_argument('--image_dir_ls', type = str, nargs ='*', default=None, help = 'Directories with images corresponding to the'
                                                                                        'labels. ')
        parser.add_argument('--style_dir_ls', type = str, nargs ='*', default=None, help = 'Directories with images corresponding to the'
                                                                                        'labels. ')
        parser.add_argument('--label_dir', type = str, help = "Directory where the labels niis are stored. ")
        parser.add_argument('--lesion_dir', type=str, help="Directory where the lesion niis are stored. ")
        parser.add_argument('--modalities', type = str, nargs ='*', help = 'Modalities corresponding to the directories.')
        parser.add_argument('--unseen_styles', type=str, nargs='*', help='Unseen styles / datasets contained in the style dirs.')
        parser.add_argument('--datasets', type=str, nargs='*',
                            help='Datasets (styles) that are part of the data provided in style_dir, image_dir.')
        parser.add_argument('--batchSize', type=int, default=10, help='Size of the batch that will be used.')
        parser.add_argument('--sample_mod', type = str, default = 'all', help='Option for modality sampling: all, or a specific modality ')
        parser.add_argument('--use_mod_disc', type = bool, help='Use modality discriminator to asses the modality-quality of the image. ')
        parser.add_argument('--mod_disc_path', type = str, help='Path to the modality-dataset discrimination network pth file. ')
        parser.add_argument('--style_translation', type = bool, help = 'Whether to use style translation or not. ' )
        parser.add_argument('--use_gpu', type=bool,
                            help='Whether to use GPU or CPU')
        parser.add_argument('--style_translation_mode', type = str, default = 'all', help = 'whether we look for alternative'
                                                                                            'styles in self.datasets (seen),'
                                                                                            'styles in self.unseen_datasets'
                                                                                            '(unseen) or all if style translation'
                                                                                            'is active.')
        parser.add_argument('--samples_per_subj', type=int, default = 1, help = "Number of samples to draw for one subject.")
        parser.add_argument('--style_mask_dir', type = str, default = None, help = 'Directory where the labels corresponding to'
                                                                               'the styles are. Only necessary if skull strip is'
                                                                               'used.')
        parser.add_argument('--sample_from', type = str, default = None, help = "Directory from which you want to base new slices."
                                                                                "Must be a directory with png files (NAME_slice.png")
        parser.add_argument('--skullstrip', type = bool, default = False, help = 'Whether to skullstrip or not')
        if self.style_translation_mode not in ['all', 'seen', 'unseen']:
            ValueError("Style translation mode must be either all, seen or unseen.")

        if self.style_dir_ls is None and self.image_dir_ls is None:
            ValueError("Either style_dir or image_dir must be populated.")
        if (self.style_dir_ls is not None and len(self.style_dir_ls)!=
            len(self.modalities)) or (self.image_dir_ls is not None and len(self.image_dir_ls)!=
                                      len(self.modalities)):
            ValueError("The length of the image or style directory must match that of the modality directory."
                       )

        if len(self.lesion_ids_ind) != len(self.lesion_ids_names):
            ValueError("The lesion indices and lesion names must have the same length.")
        if self.sample_mod != 'all' and self.sample_mod != 'random':
            self.sample_mod = self.sample_mod.split("-")

        self.image_dir = {}
        if self.style_dir is not None:
            if self.skullstrip:
                if self.style_mask is None:
                    ValueError("If skull-strip is active, and style_dir is used, style_mask must be populated!"
                               "Otherwise leave style_dir as none or remove skullstrip flag!")

        return parser
