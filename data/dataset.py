import monai
from data.dataset_utils import findSimilarName
import os
import numpy as np
import tqdm
from monai.data.dataset import PersistentDataset, Dataset
import pandas as pd
from data.dataset_utils import getExtension
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import threading
import torch.distributed as dist
from monai.data import partition_dataset, ThreadDataLoader
import torch

class Spade3DSet():

    def __init__(self, opt):

        self.opt = opt
        self.modalities = opt.sequences
        self.fix_seq = opt.fix_seq
        if opt.mode == 'train':
            self.use_ddp = opt.use_ddp
        else:
            self.use_ddp = False
        self.style_extension = None
        self.cut = opt.cut
        self.non_corresponding_dirs = opt.non_corresponding_dirs
        self.non_corresponding_ims = opt.non_corresponding_ims
        self.style_slice_consistency = opt.style_slice_consistency


        # Image, label and styles
        self.data_dict, self.modalities_here, self.datasets = self.readTSV(opt.data_dict,
                                                                           keywords=['label'] + self.modalities,
                                                                           is_styles=False, )
        # Styles
        if self.opt.style_dict is not None and self.non_corresponding_dirs:
            self.style_dict, self.modalities_here, self.datasets = self.readTSV(opt.style_dict,
                                                                                keywords=['label'] + self.modalities)
        else:
            self.style_dict = self.data_dict
            self.style_extension = '.nii.gz'

        if len(self.modalities_here) == 0:
            ValueError("No modality was found on image dir or style dir. Check your tsv files."
                       "There should be at least one out of: %" %",".join(self.modalities))

        self.skullstrip = opt.skullstrip
        if 'label' not in self.style_dict[0].keys() and self.skullstrip:
            ValueError("Key label is not present in style_dict, despite skullstrip being true."
                       "Either set skullstrip to False or provide labels.")

        # Set up style directory.
        if self.non_corresponding_dirs and self.non_corresponding_ims:
            self.non_corresponding_ims = False
            Warning("Flag non_corresponding_ims is set to True, but we are setting it to False, because"
                    "non_corresponding_dirs is False.")
        self.mode = opt.mode
        self.crop_size = opt.crop_size
        self.chunk_size = opt.chunk_size
        if self.chunk_size is None:
            self.chunk_size = opt.crop_size[-1]
        self.max_size = opt.max_dataset_size
        self.cache_dir = None

        if self.mode == 'train':
            # Handle validation items
            if opt.data_dict_val is not None:
                self.data_dict_val_raw, mod_vals, _ = self.readTSV(opt.data_dict_val, keywords=['label'] + self.modalities,
                                                            is_styles=False)
                if False in [True for v in mod_vals if v in self.modalities_here]:
                    raise Warning("Not all the modalities present in the validation dictionary are present in the "
                                  "training dicrionaries.")
            else:
                self.data_dict_val_raw = None

            # Handle this well
            # Styles
            if not self.non_corresponding_dirs:
                if self.opt.style_dict_val is None:
                    self.style_dict_val = None
                else:
                    self.style_dict_val, _, _ = self.readTSV(opt.style_dict_val,
                                                          keywords=['label'] + self.modalities)
            else:
                self.style_dict_val = None
            self.data_dict_train, self.data_dict_val = self.retrieveMONAIdicts(shuffle=opt.shuffle, perc_val=opt.perc_val)
            if self.opt.cache_type == 'cache':
                self.cache_dir = opt.cache_dir
        else:
            self.data_dict_val_raw = None
            self.style_dict_val = None
            self.data_dict_test, _ = self.retrieveMONAIdicts(shuffle=opt.shuffle, perc_val=0.0)
            self.data_dict_val = None

    def getDatasets(self):
        lock = threading.Lock()
        lock.acquire()
        if self.mode == 'train':
            transforms_train, transforms_val = self.getTransforms()
            if self.opt.cache_type == 'cache':
                if self.use_ddp:
                    data_dict_val = self.ddpDataDicts(self.data_dict_val)
                    data_dict_train = self.ddpDataDicts(self.data_dict_train)
                    val_set = PersistentDataset(data_dict_val, transform=transforms_val,
                                                cache_dir=self.cache_dir)
                    train_set = PersistentDataset(data_dict_train, transform=transforms_train,
                                                cache_dir=self.cache_dir)
                else:
                    val_set = PersistentDataset(self.data_dict_val, transform=transforms_val,
                                                cache_dir=self.cache_dir)
                    train_set = PersistentDataset(self.data_dict_train, transform=transforms_train,
                                                  cache_dir=self.cache_dir)
            else:
                if self.use_ddp:
                    data_dict_val = self.ddpDataDicts(self.data_dict_val)
                    data_dict_train = self.ddpDataDicts(self.data_dict_train)
                    val_set = Dataset(data_dict_val, transform=transforms_val)
                    train_set = Dataset(data_dict_train, transform=transforms_train)
                else:
                    val_set = Dataset(self.data_dict_val, transform=transforms_val)
                    train_set = Dataset(self.data_dict_train, transform=transforms_train)
            lock.release()
            return train_set, val_set
        else:
            transforms_test = self.getTransforms()
            if self.opt.cache_type == 'cache':
                if self.use_ddp:
                    data_dict_test = self.ddpDataDicts(self.data_dict_test)
                    test_set = PersistentDataset(data_dict_test, transform=transforms_test,
                                                      cache_dir=self.cache_dir)
                else:
                    test_set = PersistentDataset(self.data_dict_test, transform=transforms_test,
                                                      cache_dir=self.cache_dir)
            else:
                if self.use_ddp:
                    data_dict_test = self.ddpDataDicts(self.data_dict_test)
                    test_set = Dataset(data_dict_test, transform=transforms_test)
                else:
                    test_set = Dataset(self.data_dict_test, transform=transforms_test)
            lock.release()
            return test_set

    def __len__(self):
        if self.mode == 'train':
            return len(self.data_dict_train)
        else:
            return len(self.data_dict_test)

    def readTSV(self, file, keywords, is_styles = False, add_dataset = False):

        '''
        Reads TSV file containing the label / images pairs or the style labels / style images pairs.
        :param file: path to TSV file
        :param keywords: keywords (modalities) that you want included
        :param is_styles: whether this is a styles TSV or the input label and images TSV. In styles, the
        presence of keyword label isnt mandatory.
        :return:
        '''

        # Which modalities are here, which aren't
        df = pd.read_csv(file, sep="\t")
        data_dicts = []
        datasets = []
        if "label" not in list(df.columns) and not is_styles:
            ValueError("Label needs to be provided in the TSV")
        columns = list(df.columns)
        if "dataset" in list(df.columns):
            add_dataset = True
        modalities = []
        for key in keywords:
            if key in columns:
                if key != 'label':
                    modalities.append(key)

        for index, row in df.iterrows():
            out_dict = {}
            for col_ind, col in enumerate(row):
                if columns[col_ind] not in keywords:
                    continue
                if col.lower() == "none":
                    out_dict[columns[col_ind]] = None
                else:
                    out_dict[columns[col_ind]] = col
            if add_dataset:
                datasets.append(row['dataset'])
            data_dicts.append(out_dict)

        return data_dicts, modalities, list(np.unique(datasets))

    def retrieveMONAIdicts(self, shuffle = False, perc_val = 0.15):

        output = []
        output_val = []
        if self.data_dict_val_raw is None:
            if perc_val == 0.0:
                data_dict_val = []
                data_dict_train = self.data_dict[:min(len(self.data_dict), self.max_size)]
            else:
                data_dict_val = self.data_dict[:int(len(self.data_dict)*perc_val)]
                data_dict_train = self.data_dict[int(len(self.data_dict)*perc_val):]
                data_dict_train = data_dict_train[:min(self.max_size, len(data_dict_train))]
        else:
            data_dict_val = self.data_dict_val_raw
            data_dict_train = self.data_dict[:min(self.max_size, len(self.data_dict))]

        if shuffle:
                np.random.shuffle(data_dict_train)

        if self.non_corresponding_dirs:
            # We create a list of dictionaries with 1) labels 2) images
            if self.style_dict_val is None:
                style_dict_val = self.style_dict[:int(len(self.style_dict)*perc_val)]
                style_dict_train = self.style_dict[int(len(self.style_dict)*perc_val):]
            else:
                style_dict_val = self.style_dict_val
                style_dict_train = self.style_dict
        else:
            style_dict_val = data_dict_val
            style_dict_train = data_dict_train

        if self.fix_seq is not None:
            modalities = [self.fix_seq]
        else:
            modalities = self.modalities_here

        # We create list of data as a function of which elements have which modalities
        # We want to make sure that all the labels are forwarded. If the modality isn't present, then we
        # re-use some elements to make sure that the un-used one is, in the end, pushed as well. IE:
        # DICT = [{L1, T1, T2}, {L2, T1}, {L3, T1, T2}, {L4, T1}]
        # We want to sample T1, then T2, then T1, then T2, so:
        # (L1, T1), (L3, T2), (L2, T1), (L1, T2), (L4, T1)

        style_dict_train_per_mod = {}
        data_dict_train_per_mod = {}
        counter_data = {}
        for mod in modalities:
            style_dict_train_per_mod[mod] = [i for ind, i in enumerate(style_dict_train) if i[mod] is not None]
            if len(style_dict_train_per_mod[mod]) == 0:
                ValueError("When looking for the styles on the training set, modality %s has no representatives."
                           "Look at your TSV file and at your percentage validation, and ensure that there is at"
                           "least one element containing each of the modalities in each split; the files in "
                           "the TSV aren't shuffled, so consider rearranging them to fullfill this requirement,"
                           "or increase your percentage of validation.")
            if not self.non_corresponding_dirs:
                data_dict_train_per_mod[mod] = [i for ind, i in enumerate(data_dict_train) if i[mod] is not None]
                if len(data_dict_train_per_mod[mod]) == 0 and not self.non_corresponding_dirs:
                    ValueError("When looking for the styles on the training set, modality %s has no representatives."
                               "Look at your TSV file and at your percentage validation, and ensure that there is at"
                               "least one element containing each of the modalities in each split; the files in "
                               "the TSV aren't shuffled, so consider rearranging them to fullfill this requirement,"
                               "or increase your percentage of validation.")
                if shuffle:
                    np.random.shuffle(data_dict_train_per_mod[mod])
            else:
                try:
                    data_dict_train_per_mod[mod] = [i for ind, i in enumerate(data_dict_train) if i[mod] is not None]
                    if shuffle:
                        np.random.shuffle(data_dict_train_per_mod[mod])
                except:
                    data_dict_train_per_mod[mod] = None
                    if self.mode == 'train':
                        ValueError("If self.non_corresponding_dirs is True, you still need to have ground truth images if"
                               "you are in train mode: data_dict should have ground truth images!")
            counter_data[mod] = 0

        style_dict_val_per_mod = {}
        data_dict_val_per_mod = {}
        counter_data_val = {}
        for mod in modalities:
            style_dict_val_per_mod[mod] = [i for ind, i in enumerate(style_dict_val) if i[mod] is not None]
            if not self.non_corresponding_dirs:
                data_dict_val_per_mod[mod] = [i for ind, i in enumerate(data_dict_val) if i[mod] is not None]
            else:
                if len(data_dict_val) != 0 and mod in data_dict_val[0].keys():
                        data_dict_val_per_mod[mod] = [i for ind, i in enumerate(data_dict_val) if i[mod] is not None]
                else:
                    data_dict_val_per_mod[mod] = None
            if shuffle and data_dict_val_per_mod[mod] is not None:
                np.random.shuffle(data_dict_val_per_mod[mod])
            counter_data_val[mod] = 0
            if len(data_dict_val) > 0:
                if len(data_dict_val_per_mod[mod]) == 0 and not self.non_corresponding_dirs:
                    ValueError("When looking for the styles on the training set, modality %s has no representatives."
                               "Look at your TSV file and at your percentage validation, and ensure that there is at"
                               "least one element containing each of the modalities in each split; the files in "
                               "the TSV aren't shuffled, so consider rearranging them to fullfill this requirement,"
                               "or increase your percentage of validation.")
                if len(style_dict_val_per_mod[mod]) == 0:
                    ValueError(
                        "When looking for the styles on the training set, modality %s has no representatives."
                        "Look at your TSV file and at your percentage validation, and ensure that there is at"
                        "least one element containing each of the modalities in each split; the files in "
                        "the TSV aren't shuffled, so consider rearranging them to fullfill this requirement,"
                        "or increase your percentage of validation.")

        # Training dictionary
        to_process = {'train': {'data_dict': data_dict_train, 'data_dict_per_mod': data_dict_train_per_mod,
                                'style_dict': style_dict_train, 'style_dict_per_mod': style_dict_train_per_mod,
                                'counter_data': counter_data, 'output': output}
                      }
        if len(data_dict_val) > 0:
            to_process['validation'] = {'data_dict': data_dict_val, 'data_dict_per_mod': data_dict_val_per_mod,
                                        'style_dict': style_dict_val, 'style_dict_per_mod': style_dict_val_per_mod,
                                        'counter_data': counter_data_val, 'output': output_val}


        for key, val in to_process.items():
            data_train_ctr = {}
            for el in val['data_dict']:
                data_train_ctr[el['label']]= False
            counter = 0
            while False in data_train_ctr.values():
                output_dict = {}
                if self.mode == 'train':
                    modality = modalities[counter % len(modalities)]  # Chosen modalities
                    element_data = val['data_dict_per_mod'][modality][val['counter_data'][modality]]
                    output_dict['label'] = element_data['label']
                    output_dict['image'] = element_data[modality]
                    if self.non_corresponding_dirs or self.non_corresponding_ims:
                        element_style = np.random.choice(val['style_dict_per_mod'][mod])
                        output_dict['style_image'] = element_style[modality]
                        if 'label' in element_style.keys():
                            output_dict['style_mask'] = element_style['label']
                        else:
                            output_dict['style_mask'] = None
                        if self.style_slice_consistency and getExtension(output_dict['style_mask']) != '.nii.gz':
                            element_style = np.random.choice(val['style_dict_per_mod'][mod])
                            output_dict['style_image_scextra'] = element_style[modality]
                            if 'label' in element_style.keys():
                                output_dict['style_mask_scextra'] = element_style['label']
                            else:
                                output_dict['style_mask_scextra'] = None

                    else:
                        output_dict['style_image'] = element_data[modality]
                        output_dict['style_mask'] = element_data['label']
                        if getExtension(output_dict['style_mask']) != '.nii.gz' and self.style_slice_consistency:
                            output_dict['style_image_scextra'] = element_data[modality]
                            output_dict['style_mask_scextra'] = element_data['label']

                    if self.style_extension is None:
                        self.style_extension = getExtension(output_dict['style_mask'])

                    val['counter_data'][modality] = val['counter_data'][modality] + 1

                    if val['counter_data'][modality] == len(val['data_dict_per_mod'][modality]):
                        val['counter_data'][modality] = 0
                else:
                    # Test mode: image is not a requirement, we dont need to ensure modality balance either.
                    element_data = val['data_dict'][counter]
                    output_dict['label'] = element_data['label']

                    # Modalities available in this element and modalities
                    possible_modalities = [el for el in element_data.keys() if (el in modalities and element_data[el] \
                                                       is not None)]

                    if len(possible_modalities) == 0 and not self.non_corresponding_ims and not self.non_corresponding_dirs:
                        ValueError("Element %s does not contain any relevant modality. Either set self.non_corresponding_ims"
                                   "or self.non_corresponding_dirs to 1")
                        counter += 1
                        data_train_ctr[element_data['label']] = True
                        continue
                    else:
                        modality = np.random.choice(possible_modalities)
                        if self.non_corresponding_ims:
                            element_style = np.random.choice(val['style_dict_per_mod'][modality])
                        else:
                            element_style = element_data
                        output_dict['image'] = output_dict['style_image'] = element_style[modality]
                        output_dict['style_mask'] = element_style['label']
                        if self.style_slice_consistency and getExtension(output_dict['style_mask']) != '.nii.gz':
                            output_dict['style_image_scextra'] = element_style[modality]
                            output_dict['style_mask_scextra'] = element_style['label']
                    if self.style_extension is None:
                        self.style_extension = getExtension(output_dict['style_mask'])

                counter += 1
                data_train_ctr[element_data['label']] = True
                output_dict['this_seq'] = modality
                val['output'].append(output_dict)

        if len(data_dict_val) == 0:
            to_process['validation'] = {'output': []}
        return to_process['train']['output'], to_process['validation']['output']

    def ddpDataDicts(self, data_dict):

        if dist.is_initialized():
            print(dist.get_rank())
            print(dist.get_world_size())
            return partition_dataset(
                data=data_dict,
                num_partitions=dist.get_world_size(),
                shuffle=False,
                seed=0,
                drop_last=False,
                even_divisible=True

            )[dist.get_rank()]

    def getTransforms(self):

        if self.data_dict_val is None or self.data_dict_train is None or self.style_extension is None:
            ValueError("Cannot call function getTransforms if the data dictionaries aren't initialised."
                       "Call self.reetrieveMONAIdicts.")

        # Different keys to be loaded: label, image, style_image and style_mask.
        # Depending on extension, style_image and style_mask will be loaded with Nifti or NPZ
        # If NPZ: we consider the slices are 2D, and therefore don't need random slicing.
        # Otherwise, crop in the center and then selection of one slice will be applied on the nifti volume to get
        # the 2D slice.

        if self.chunk_size is not None:
            roi_randcrop = [-1, -1, self.chunk_size]
        if self.cut == 'a':
            roi_size = [-1, -1, 1]
            roi_crop = [-1, -1, self.crop_size[-1]//3]
            squeeze_dim = -1
        elif self.cut == 'c':
            roi_size = [-1, 1, -1]
            roi_crop = [-1, self.crop_size[-2]//3, -1]
            squeeze_dim = -2
        else:
            roi_size = [1, -1, -1]
            roi_crop = [self.crop_size[-3]//3, -1, -1]
            squeeze_dim = -3

        # Loading
        transforms = []

        # This is horribly coded... sorry! In a rush.
        if self.style_extension == '.nii.gz':
            if self.non_corresponding_dirs and self.mode != 'train':
                keys_volume = ['label']
                keys_slice = ['image', 'style_mask', 'style_image']
            else:
                keys_volume = ['label', 'image']
                keys_slice = ['style_mask', 'style_image']
            transforms.append(monai.transforms.LoadImaged(keys = ['image', 'label', 'style_image', 'style_mask'])) # Niftis
            transforms.append(monai.transforms.AddChanneld(keys=['style_image', 'image']))
            transforms.append(monai.transforms.AsChannelFirstd(keys=['label', 'style_mask'], channel_dim=-1))
            transforms.append(monai.transforms.CenterSpatialCropd(roi_size=roi_crop, keys = keys_slice))
            if self.style_slice_consistency:
                transforms.append(monai.transforms.CopyItemsd(keys=['style_image'], names=['style_image_scextra']))
                transforms.append(monai.transforms.CopyItemsd(keys=['style_mask'], names=['style_mask_scextra']))

            if self.style_slice_consistency:
                transforms.append(monai.transforms.RandSpatialCropd(max_roi_size=roi_size, roi_size=roi_size,
                                                                    keys = ['style_image_scextra', 'style_mask_scextra'],))
            transforms.append(monai.transforms.RandSpatialCropd(max_roi_size=roi_size, roi_size=roi_size,
                                                                keys = keys_slice))
            if self.style_slice_consistency:
                keys_slice.append("style_mask_scextra")
                keys_slice.append("style_image_scextra")


            transforms.append(monai.transforms.SqueezeDimd(dim = squeeze_dim, keys = keys_slice))
            transforms.append(monai.transforms.CenterSpatialCropd(keys=keys_volume, roi_size=self.crop_size))
            transforms.append(monai.transforms.SpatialPadd(keys=keys_volume, spatial_size=self.crop_size,
                                                           method='symmetric'))
            if self.chunk_size is not None:
                transforms.append(monai.transforms.RandSpatialCropd(max_roi_size=roi_randcrop, roi_size=roi_randcrop,
                                                                    keys = keys_volume))
            transforms.append(
                monai.transforms.SpatialPadd(keys=keys_slice, spatial_size=self.crop_size[:-1],
                                             method='symmetric'))
            transforms.append(monai.transforms.Lambdad(keys=[ks for ks in keys_slice if 'mask' in ks], func=lambda l: np.concatenate(
                [np.expand_dims(1 - np.sum(l[1:, ...], 0), 0), l[1:, ...]], 0)))
            transforms.append(monai.transforms.Lambdad(keys=['label'], func=lambda l: np.concatenate(
                [np.expand_dims(1 - np.sum(l[1:, ...], 0), 0), l[1:, ...]], 0)))
            transforms.append(monai.transforms.CenterSpatialCropd(keys=keys_slice,
                                                                  roi_size=self.crop_size[:-1]))

        else:
            if self.non_corresponding_dirs and self.mode != 'train':
                transforms.append(monai.transforms.LoadImaged(keys=['label'])) # Niftis
                transforms.append(monai.transforms.LoadImaged(keys=['style_image', 'image'],
                                                              reader='numpyreader', npz_keys=('img')))  # Npz
                transforms.append(monai.transforms.LoadImaged(keys=['style_mask'], reader='numpyreader', npz_keys = ('label')))
                transforms.append(monai.transforms.AddChanneld(keys=['style_image'],))
                transforms.append(monai.transforms.AddChanneld(keys=['image']))
                transforms.append(monai.transforms.EnsureChannelFirstd(keys=['label'], channel_dim=-1))
                transforms.append(monai.transforms.EnsureChannelFirstd(keys=['style_mask'], channel_dim=-1))
                transforms.append(monai.transforms.CenterSpatialCropd(keys=['label'], roi_size=self.crop_size))
                transforms.append(monai.transforms.SpatialPadd(keys=['label'], spatial_size=self.crop_size,
                                                               method='symmetric'))
                if self.chunk_size is not None:
                    transforms.append(
                        monai.transforms.RandSpatialCropd(max_roi_size=roi_randcrop, roi_size=roi_randcrop,
                                                          keys=['label']))
                transforms.append(
                    monai.transforms.SpatialPadd(keys=['style_image','image'], spatial_size=self.crop_size[:-1],
                                                 method='symmetric'))
                transforms.append(
                    monai.transforms.SpatialPadd(keys=['style_mask'], spatial_size=self.crop_size[:-1],
                                                 method='symmetric', mode = 'constant', constant_values = 1.0))
                transforms.append(monai.transforms.Lambdad(keys=['label'], func=lambda l: np.concatenate(
                    [np.expand_dims(1 - np.sum(l[1:, ...], 0), 0), l[1:, ...]], 0)))
                transforms.append(monai.transforms.CenterSpatialCropd(keys=['style_image', 'style_mask', 'image'],
                                                                      roi_size=self.crop_size[:-1]))
            else:
                transforms.append(
                    monai.transforms.LoadImaged(keys=['image', 'label']))  # Niftis
                transforms.append(monai.transforms.LoadImaged(keys=['style_image'], reader='numpyreader', npz_keys=('img')))  # Npz
                transforms.append(monai.transforms.LoadImaged(keys=['style_mask'], reader='numpyreader', npz_keys=('label')))  # Npz
                if self.style_slice_consistency:
                    transforms.append(monai.transforms.LoadImaged(keys=['style_image_scextra'], reader='numpyreader',
                                                                  npz_keys=('img')))  # Npz
                    transforms.append(monai.transforms.LoadImaged(keys=['style_mask_scextra'], reader='numpyreader',
                                                                  npz_keys=('label')))  # Npz
                    transforms.append(monai.transforms.AddChanneld(keys=['style_image_scextra']))
                    transforms.append(monai.transforms.AddChanneld(keys=['style_mask_scextra']))

                transforms.append(monai.transforms.AddChanneld(keys=['style_image']))
                transforms.append(monai.transforms.AddChanneld(keys=['image']))
                transforms.append(monai.transforms.EnsureChannelFirstd(keys=['label'], channel_dim=-1))
                transforms.append(monai.transforms.EnsureChannelFirstd(keys=['style_mask'], channel_dim=-1))
                transforms.append(monai.transforms.CenterSpatialCropd(keys=['image', 'label'], roi_size=self.crop_size))
                transforms.append(monai.transforms.SpatialPadd(keys=['image', 'label'], spatial_size=self.crop_size,
                                                               method='symmetric'))
                if self.chunk_size is not None:
                    transforms.append(
                        monai.transforms.RandSpatialCropd(max_roi_size=roi_randcrop, roi_size=roi_randcrop,
                                                          keys=['image', 'label']))
                if self.style_slice_consistency:
                    keys_slice = ['style_image', 'style_mask', 'style_mask_scextra', 'style_image_scextra']
                else:
                    keys_slice = ['style_image', 'style_mask',]
                transforms.append(
                    monai.transforms.SpatialPadd(keys=keys_slice, spatial_size=self.crop_size[:-1],
                                                 method='symmetric'))
                transforms.append(monai.transforms.Lambdad(keys=[ks for ks in keys_slice if 'mask' in ks], func=lambda l: np.concatenate(
                    [np.expand_dims(1 - np.sum(l[1:, ...], 0), 0), l[1:, ...]], 0)) if l.shape[0] > 1 else l)
                transforms.append(monai.transforms.Lambdad(keys=['label'], func=lambda l: np.concatenate(
                    [np.expand_dims(1 - np.sum(l[1:, ...], 0), 0), l[1:, ...]], 0)))
                transforms.append(monai.transforms.CenterSpatialCropd(keys=keys_slice,
                                                                      roi_size=self.crop_size[:-1]))

        # Augmenting
        aug_transforms = []
        if self.opt.mode == 'train' and self.opt.augment:
            aug_transforms.append(monai.transforms.RandBiasFieldd(coeff_range=(0, 0.005), prob=0.33, keys=[ks for ks in keys_slice if 'image' in ks]),
                              )
            aug_transforms.append(monai.transforms.RandAdjustContrastd(gamma=(0.9, 1.15), prob=0.33, keys=[ks for ks in keys_slice if 'image' in ks]))
            aug_transforms.append(monai.transforms.RandGaussianNoised(prob=0.33, mean=0.0,
                                                    std=np.random.uniform(0.005, 0.015),
                                                                  keys=[ks for ks in keys_slice if 'image' in ks]))
            aug_transforms.append(monai.transforms.RandAffined(
                rotate_range=[-0.05, 0.05],
                shear_range=[0.001, 0.05],
                scale_range=[0, 0.05],
                padding_mode='zeros',
                mode='nearest',
                prob=0.33,
                keys = ['label', 'image']))

        # transforms.Lambdad(keys=['label'], func = lambda l: np.concatenate(
        #     [np.expand_dims(1-np.sum(l[1:, ...], 0),0), l[1:,...]], 0)),

        end_transforms = []
        end_transforms.append(monai.transforms.CopyItemsd(keys=['label'], names=['label_channel']))
        end_transforms.append(monai.transforms.CopyItemsd(keys=['style_mask'], names=['style_mask_channel']))
        if self.style_slice_consistency:
            end_transforms.append(monai.transforms.CopyItemsd(keys=['style_mask_scextra'], names=['style_mask_scextra_channel']))

        if self.non_corresponding_dirs and self.mode != "train":
            mask_4_image_key = 'style_mask_channel'
        else:
            mask_4_image_key = 'label_channel'

        if self.opt.skullstrip:
            end_transforms.append(monai.transforms.Lambdad(keys = ['label_channel'],
                                                           func = lambda l: np.stack([l[0, ...] < 0.35]*1, 0)))
            end_transforms.append(monai.transforms.Lambdad(keys = ['style_mask_channel'],
                                                           func = lambda l: np.stack([l[0, ...] < 0.35]*1, 0)))
            end_transforms.append(monai.transforms.MaskIntensityd(keys = ['image'], mask_key=mask_4_image_key))
            end_transforms.append(monai.transforms.MaskIntensityd(keys=['style_image'], mask_key='style_mask_channel'))
            if self.style_slice_consistency:
                end_transforms.append(monai.transforms.Lambdad(keys=['style_mask_scextra_channel'],
                                                               func=lambda l: np.stack([l[0, ...] < 0.35] * 1, 0)))
                end_transforms.append(
                    monai.transforms.MaskIntensityd(keys=['style_image_scextra'], mask_key='style_mask_scextra_channel'))
        end_transforms.append(monai.transforms.NormalizeIntensityd(keys=['image']))
        end_transforms.append(monai.transforms.NormalizeIntensityd(keys=['style_image']))
        if self.style_slice_consistency:
            end_transforms.append(monai.transforms.NormalizeIntensityd(keys=['style_image_scextra']))
            end_transforms.append(monai.transforms.ToTensord(keys=['style_image_scextra', 'style_mask_scextra']))
        end_transforms.append(monai.transforms.ToTensord(keys=['style_image', 'image', 'label', 'style_mask'],
                                                         ),
                              )

        val_transforms = monai.transforms.Compose(transforms + end_transforms)
        if self.opt.augment:
            train_transforms = monai.transforms.Compose(transforms + aug_transforms + end_transforms)

        if self.mode == 'train':
            return train_transforms, val_transforms
        else:
            return val_transforms

    def resetSeqs(self, seqs):
        if type(seqs) is list:
            self.modalities = seqs
        else:
            self.fix_seq = seqs
        if self.mode == 'train':
            self.data_dict_train, self.data_dict_val = self.retrieveMONAIdicts(shuffle=self.opt.shuffle, perc_val=0.0)
            train_set, val_set = self.getDatasets()
            return train_set, val_set
        else:
            self.data_dict_test, _ = self.retrieveMONAIdicts(shuffle=False, perc_val=0.0)
            test_set = self.getDatasets()
            return test_set