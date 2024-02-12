import torch
import numpy
import os
import csv
import numpy as np
import shutil
import nibabel as nib
from typing import List, Optional, Tuple, Union
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
from monai.data import partition_dataset

LESION_CHANNELS = {"wmh": 6, "tumour": 7, "edema": 8, "gdtumour": 9}
DATASETS = ['SABRE', 'BRATS-TCIA', 'ABIDE', 'BRATS-OTHER', 'ADNI', 'ADNI2', 'OASIS', 'OAS',
            ]
def post_process_os(data_i):
    """
    From a SPADE Dataset item framework, extract the strings from this_seq and other_seqs to put them right after auto-collate fn call
    :param data_i: Data item obtained from pix2pix dataset
    :return:
    """

    if 'this_seq' in data_i.keys():
        var = data_i['this_seq']
        if not type(var) == str:
            data_i['this_seq'] = var[0]
    return data_i

def clear_data(trainer, data_i):

    """
    Detaches some data from the GPU to ensure that the training is possible
    :param data_i:
    :return:
    """

    data_i['image'] = data_i['image'].detach().cpu()
    if trainer is not None:
        if trainer.generated is not None:
            trainer.generated = trainer.generated.detach().cpu()

def findSimilarName(original, directory, slice_index=-1, extension='.png', return_all = False, additional_keywords = []):
    if type(directory) is str:
        all_files = os.listdir(directory)
    else:
        all_files = directory
    keywords = ["sub", "ses", "SUB", "SES",]+additional_keywords
    root = original.replace(extension, "")
    root = root.split("/")[-1] # If it's a path, we take it out
    if return_all:
        return_list = []
    for f_sup in all_files:
        f_path = "/".join(f_sup.split("/")[:-1])
        f = f_sup.replace(extension, "")
        f_sp = f.split("_")
        keys = []
        for key in keywords:
            positives_targ = [sp for sp in f_sp if key in sp]
            if len(positives_targ) == 0:
                continue
            else:
                positives_targ = positives_targ[0]
                ori_sp = root.split("_")
                positives_ori = [sp for sp in ori_sp if key in sp]
                if len(positives_ori) == 0:
                    continue
                else:
                    positives_ori = positives_ori[0]
                    if positives_targ == positives_ori:
                        keys.append(True)
                    else:
                        keys.append(False)
                        break
        # Now we compare the slice number if applicable
        if slice_index is not None:
            slice = f_sp[slice_index]
            slice_ori = root.split("_")[-1]
            if slice != slice_ori:
                keys.append(False)
        if False not in keys:
            if return_all:
                return_list.append(os.path.join(f_path,f+extension))
            else:
                return os.path.join(f_path, f + extension)
    if return_all:
        if len(return_list) > 0:
            return return_list
        else:
            return  None
    else:
        return None

def create_datatext_source_image_gen(in_dirs, out_file, add_dataset = False, new_source = None,
                                     trvate = False, perc_val = 0.05, perc_test = 0.1,
                                     overwrite = False, additional_keywords = [],
                                     slice_index = None,
                                     filters=None):
    '''

    :param in_dirs: list of dictionaries, containing keywords representing the type of file embedded in the
    directory: i.e. [{label: path_to_labels, T1: path_to_T1s, FLAIR: path_to_FLAIRs}]
    :param out_file: filepath (including .tsv extension)where you want to store the dicts
    :param add_dataset: whether to add the dataset name as a last column. Make sure the datasets handled are in
    the above list DATASETS. Otherwise, add them.
    :param new_source: dict,  if you want to map a part of the path to another (for mounted volumes)
    :param trvate: if True, train / test and validation splits will be created.
    :param perc_val: percentage of rows devoted to validation
    :param perc_test: percentage of rows devoted to test
    :param overwrite: if the file exists, overwrite = True overwrites it
    :param additional_keywords: if there is some keyword (str) that you want to use as match to find equivalent
    filenames in the different directories, pass it a list of string. For more information, see function findSimilarName.
    :param slice_index: If these aren't whole volumes, but slices, slice index is the index of the filename split
    by "_" where the slice number is (i.e. : File_something_something_[slice_no].png > slice_index = -1).
    :param filters: if not None, it needs to be a dictionary with key train and test-val, pointing to a txt with
    list of volumes to take into account for both splits. ignored if trvate is false
    :return:
    '''

    if 'label' not in in_dirs[0].keys():
        ValueError("Key label is a required data piece.")

    row_names = ['label']
    # Modalities: there can be some missing for some directories.
    for dir in in_dirs:
        for key in dir.keys():
            if key != 'label' and key not in row_names:
                row_names.append(key)
    if add_dataset:
        row_names.append('dataset')

    rows = []
    for dir in in_dirs:
        # For each directory set, we note the label, and its corresponding modality images if they exist.
        all_labels = os.listdir(dir['label'])
        for label in all_labels:
            if new_source is not None:
                dir_new = dir['label']
                for key, val in new_source.items():
                    dir_new = dir_new.replace(key, val)
            else:
                dir_new = dir['label']
            row = [os.path.join(dir_new, label)]
            for mod in row_names[1:]:
                if mod == 'dataset':
                    continue
                if mod not in dir.keys():
                    row.append("none")
                else:
                    extension = label.split(".")
                    if len(extension) > 0:
                        extension = ".".join(extension[1:])
                    else:
                        extension = extension[-1]
                    im = findSimilarName(label, dir[mod], extension="."+extension,
                                         return_all=False, additional_keywords=additional_keywords,
                                         slice_index=slice_index)
                    if im is None:
                        row.append("none")
                    else:
                        if new_source is not None:
                            dir_new = dir[mod]
                            for key, val in new_source.items():
                                dir_new=dir_new.replace(key, val)
                            row.append(os.path.join(dir_new, im))
                        else:
                            row.append(os.path.join(dir[mod], im))
            if add_dataset:
                d = get_dataset(label)
                row.append(d)
            rows.append(row)

    # We shuffle
    np.random.shuffle(rows)

    if trvate:
        if filters is None:
            rows = [[row_names] + rows[:int(len(rows)*perc_val)],
                    [row_names] + rows[int(len(rows)*perc_val): (int(len(rows)*perc_val)+int(len(rows)*perc_test))],
                    [row_names] + rows[(int(len(rows)*perc_val)+int(len(rows)*perc_test)):]]
        else:
            with open(filters['train'], 'r') as f:
                labels_train = f.readlines()
                f.close()
            labels_train = [l.strip("\n") for l in labels_train]
            with open(filters['test_val'], 'r') as f:
                labels_val_test = f.readlines()
                f.close()
            labels_val_test = [l.strip("\n") for l in labels_val_test]
            rows_train = [row for row in rows if row[0].split("/")[-1] in labels_train]
            rows_val_test = [row for row in rows if row[0].split("/")[-1] in labels_val_test]
            perc_val_new = perc_val / (perc_test + perc_val)
            rows_val = rows_val_test[:int(len(rows_val_test)*perc_val_new)]
            rows_test = rows_val_test[int(len(rows_val_test)*perc_val_new):]
            rows = [[row_names] + rows_val,
                    [row_names] + rows_test,
                    [row_names] + rows_train]

        file_names = [out_file.replace(".tsv", "_%s.tsv" %i) for i in ['validation', 'test', 'train']]
        for find, f in enumerate(file_names):
            if os.path.isfile(file_names[find]) and overwrite:
                os.remove(file_names[find])
            for row in rows[find]:
                with open(file_names[find], 'at') as f:
                    tsv_writer = csv.writer(f, delimiter='\t')
                    tsv_writer.writerow(row)
                    f.close()
    else:
        rows = [row_names] + rows
        if os.path.isfile(out_file) and overwrite:
            os.remove(out_file)
        for row in rows:
            with open(out_file, 'at') as f:
                tsv_writer = csv.writer(f, delimiter='\t')
                tsv_writer.writerow(row)
                f.close()

def filterTSV(in_dirs:list, keywords:list, in_column:str, not_clause:bool=False):

    for file in in_dirs:
        df = pd.read_csv(file, sep="\t")
        if not_clause:
            matches = df.loc[df[in_column].apply(lambda x: any(k for k in keywords if k not in x))]
            matches.to_csv(file.replace(".tsv", "_%s_notin_%s.tsv" % ("-".join(keywords),
                                                                   in_column)),
                           sep="\t",
                           index=False)
        else:
            matches = df.loc[df[in_column].apply(lambda x: any(k for k in keywords if k in x))]
            matches.to_csv(file.replace(".tsv", "_%s_in_%s.tsv" % ("-".join(keywords),
                                                                   in_column)),
                           sep="\t",
                           index=False)

def getExtension(file_path):
    if ".nii.gz" in file_path:
        return ".nii.gz"
    else:
        return ".%s" %(file_path.split("/")[-1].split(".")[-1])

def get_dataset(file_path):
    if "/" in file_path:
        f_ = file_path.split("/")[-1]
    else:
        f_ = file_path
    for d in DATASETS:
        if d in f_:
            return d

def create_datatext_source_label_gen(in_dirs: Union[list, str],
                                     out_file: str,
                                     report_folder: str = None,
                                     new_source: dict = None,
                                     trvate: bool = False,
                                     shuffle: bool = False,
                                     lesions: list = [],
                                     lesion_dim: int = -1,
                                     perc_val: float = 0.05,
                                     perc_test: float = 0.1,
                                     overwrite: bool =False,
                                     lesion_pixels: dict = None,
                                     lesion_maxes: dict = None):
    '''
    Create TSVs to train label generator.
    :param in_dirs: list containing the paths with the labels or string leading to TSV file. MUST have a 'label'
    columnm from where the labels will be picked up. If trvate is ON, specify path with "*" where the test, train,
    val are. IE: the mother files are called mylist_train.tsv, mylist_validation.tsv and mylist_test.tsv.
    Pass 'mylist_*tsv' as argument here.
    :param out_file: filename where you want to save the TSV(s) (if trvate is true, _train, _validation and _test
    will be added).
    :param report_to_lesions: path to the TXT containing mins and maxes of the lesions.
    :param new_source:
    :param trvate: if True, three files will result (with train, validation and test).
    :param shuffle: bool, whether you want to shuffle the output rows.
    :param lesions: lesions that you want to add as columns (conditionings) to the TSV files.
    :param lesion_dim: in which dimension of the label the tissue / lesions are (i.e. HxWxDx[LESIONS] >> -1)
    :param perc_val: percentage devoted to validation if trvate is on.
    :param perc_test: percentage devoted to test in trvate is on
    :param overwrite: if the out file(s) exists, delete and overwrite.
    :param lesion_pixels: for recursive calls, dictionary of dictionaries containing the number of pixels for
    each lesion label
    :param lesion_maxes: for recursive calls, dictionary  containing the minimum and maximum
    number of pixels across the data for each lesions
    :return:
    '''


    # Lesion dimensions
    couldnt_read = []
    if len(lesions) > 0 and (lesion_pixels is None and lesion_maxes is None):
        lesion_pixels = {}
        lesion_maxes = {}
        all_labels = []
        if trvate and "*" in in_dirs:
            for sub_file_ in ['train', 'validation', 'test']:
                sub_file = in_dirs.replace("*", "_%s" %sub_file_)
                with open(sub_file, 'r') as f:
                    df = pd.read_csv(sub_file, sep="\t")
                    for index, row in df.iterrows():
                        all_labels.append(row['label'])
                    f.close()
        else:
            with open(in_dirs, 'r') as f:
                df = pd.read_csv(in_dirs, sep="\t")
                for index, row in df.iterrows():
                    all_labels.append(row['label'])
            f.close()

        for l in lesions:
            lesion_maxes[l] = [1000000, 0]
        for label in tqdm(all_labels):
            lesion_pixels[label] = {}
            try:
                if ".npz" in label:
                    label_array = np.load(label)['label']
                elif ".nii.gz" in label:
                    label_array = np.asarray(nib.load(label).dataobj)
                elif ".npy" in label:
                    label_array = np.load(label)
                else:
                    ValueError("Unsupported format: .%s" % label.split(".")[-1])
            except FileNotFoundError:
                couldnt_read.append(label)
                print("Not found: %s" %label)
                continue
            for lesion in lesions:
                if lesion_dim == -1:
                    lesion_array = label_array[..., LESION_CHANNELS[lesion]]
                else:
                    lesion_array = label_array[LESION_CHANNELS[lesion], ...]
                positives = (lesion_array > 0.5).sum()
                lesion_pixels[label][lesion] = positives
                if positives > lesion_maxes[lesion][-1]:
                    lesion_maxes[lesion][-1] = positives
                if positives < lesion_maxes[lesion][0]:
                    lesion_maxes[lesion][0] = positives

    if lesion_dim not in [-1, 0,]:
        ValueError("Unsupported slicing for lesion dimensions different from -1 or 0. If you have used"
                   "positive indices for the last dimension, replace them by -1. Otherwise, add some logic"
                   "to perform the slicing according to the queried dimension below, in the area labelled"
                   "as SLICING_LESIONS.")
    if type(in_dirs) is str:
        print("Using TSV file %s as data input..." %in_dirs)
        all_labels = []
        if trvate:
            # List of lists.
            for sub_file in ['train', 'validation', 'test']:
                in_dirs_sf = in_dirs.replace("*", "_%s" %sub_file)
                create_datatext_source_label_gen(in_dirs_sf, out_file=out_file.replace(".tsv", "_%s.tsv" %sub_file),
                                                 report_folder=report_folder, new_source=new_source, trvate=False,
                                                 shuffle=shuffle, lesions = lesions, lesion_dim = lesion_dim,
                                                 overwrite=overwrite, lesion_pixels = lesion_pixels,
                                                 lesion_maxes = lesion_maxes)
            return
        else:
            with open(in_dirs, 'r') as f:
                df = pd.read_csv(in_dirs, sep="\t")
                for index, row in df.iterrows():
                    all_labels.append(row['label'])
    else:
        print("Reading file names from list of directories...")
        all_labels = [os.listdir(l_dir) for l_dir in in_dirs]

    # Check the reports if present
    maximums = {}
    if report_folder is not None:
        # Collect the maximum values of pixel per value.
        with open(report_folder, 'r') as f:
            all_lines = f.readlines()
            f.close()
        for les, les_val in LESION_CHANNELS.items():
            correct_line = [f for f in all_lines if f.split(":")[0] == les]
            if len(correct_line) > 0:
                correct_line = correct_line[0].split(":")[-1].split(";")
                maximum = int([c for c in correct_line if "max" in c][0].split("=")[-1])
                maximums[les] = maximum
            else:
                maximums[les] = None

    row_names = ['label']
    # Modalities: there can be some missing for some directories.
    for lesion in lesions:
        row_names.append(lesion)
    rows = []

    # Read labels
    progress_bar = tqdm(enumerate(all_labels), total=len(all_labels), ncols=110)
    progress_bar.set_description("Reading labels and checking lesions")
    for _, label in progress_bar:
        # If we want to replace absolute paths with relatives we use new source ({abs_path_chunk : rel_path})
        if len(lesion_pixels[label])==0:
            print("We do not add label %s because it doesn't exist or could not be found." %label)
            continue
        if new_source is not None:
            dir_label = "/".join(label.split("/")[:-1])
            label_root = label.split("/")[-1]
            for key, val in new_source.items():
                dir_label = dir_label.replace(key, val)
            row = [os.path.join(dir_label, label_root)]
        else:
            row = [label]

        # Now we check for lesions
        # if len(lesions) > 0: # If we are using conditioning, otherwise we dont care.
        #     if ".npz" in label:
        #         label_array = np.load(label)['label']
        #     elif ".nii.gz" in label:
        #         label_array = np.asarray(nib.load(label).dataobj)
        #     elif ".npy" in label:
        #         label_array = np.load(label)
        #     else:
        #         ValueError("Unsupported format: .%s" %label.split(".")[-1])
        #
        #     lesions_in = {}
        #     for lesion in lesions:
        #         # SLICING-LESIONS
        #         if lesion_dim == -1:
        #             lesion_array = label_array[..., LESION_CHANNELS[lesion]]
        #         else:
        #             lesion_array = label_array[LESION_CHANNELS[lesion], ...]
        #         if report_folder is None:
        #             if (lesion_array > 0).sum() > 0:
        #                 lesions_in[lesion] = 1.0
        #             else:
        #                 lesions_in[lesion] = 0.0
        #         else:
        #             if lesion in maximums.keys():
        #                 positives = (lesion_array > 0.5).sum()
        #                 lesions_in[lesion] = np.round(positives / maximums[lesion], 6)
        #             else:
        #                 if (lesion_array > 0).sum() > 0:
        #                     lesions_in[lesion] = 1.0
        #                 else:
        #                     lesions_in[lesion] = 0.0
        #
        #         row += [lesions_in[lesion]]
        if len(lesions) > 0:
            for lesion in lesions:
                if os.path.join(label) in lesion_pixels.keys():
                    row += [np.round( lesion_pixels[label][lesion] / lesion_maxes[lesion][-1], 6)]
        rows.append(row)

    # We shuffle
    if shuffle:
        np.random.shuffle(rows)

    print("Saving TSVs...")
    if trvate:
        rows = [[row_names] + rows[:int(len(rows)*perc_val)],
                [row_names] + rows[int(len(rows)*perc_val): (int(len(rows)*perc_val)+int(len(rows)*perc_test))],
                [row_names] + rows[(int(len(rows)*perc_val)+int(len(rows)*perc_test)):]]
        file_names = [out_file.replace(".tsv", "_%s.tsv" %i) for i in ['validation', 'test', 'train']]
        for find, f in enumerate(file_names):
            for row in rows[find]:
                if os.path.isfile(file_names[find]) and overwrite:
                    os.remove(file_names[find])
                with open(file_names[find], 'at') as f:
                    tsv_writer = csv.writer(f, delimiter='\t')
                    tsv_writer.writerow(row)
                    f.close()
    else:
        rows = [row_names] + rows
        if os.path.isfile(out_file) and overwrite:
            os.remove(out_file)
        for row in rows:
            with open(out_file, 'at') as f:
                tsv_writer = csv.writer(f, delimiter='\t')
                tsv_writer.writerow(row)
                f.close()

def rewrite_tsv_with_new_sources(in_file,
                                 out_file,
                                 new_source: dict,
                                 trvate = False,
                                 at_random = False):

   if trvate:
       for sub_file in ['train', 'validation', 'test']:
           in_dirs_sf = in_file.replace("*", "_%s" %sub_file)
           in_dirs_df = pd.read_csv(in_dirs_sf, sep = "\t")
           for col in in_dirs_df:
               for i, row_value in in_dirs_df[col].iteritems():
                   for key, val in new_source.items():
                       try:
                           if key in in_dirs_df[col][i]:
                               if at_random and np.random.uniform() > 0.5:
                                   in_dirs_df[col][i] = in_dirs_df[col][i].replace(key, val)
                               elif not at_random:
                                   in_dirs_df[col][i] = in_dirs_df[col][i].replace(key, val)
                       except:
                           pass

           in_dirs_df.to_csv(out_file.replace(".tsv", "_%s.tsv" %sub_file), sep = "\t", index=False)
   else:
       in_dirs_sf = in_file.replace("*", "_%s" % in_file)
       in_dirs_df = pd.read_csv(in_dirs_sf, sep="\t")
       for col in in_dirs_df:
           for i, row_value in in_dirs_df[col].iteritems():
               for key, val in new_source.items():
                   try:
                       if key in in_dirs_df[col][i]:
                           if at_random and np.random.uniform() > 0.5:
                               in_dirs_df[col][i] = in_dirs_df[col][i].replace(key, val)
                           elif not at_random:
                               in_dirs_df[col][i] = in_dirs_df[col][i].replace(key, val)
                   except:
                       print("Couldn't perform replacement in column %s"  %col)
       in_dirs_df.to_csv(out_file, sep = "\t", index=False,)


def ddpDataDicts(data_dict):

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