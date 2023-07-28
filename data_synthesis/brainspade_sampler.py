import monai
import numpy as np
import torch
import os
import json
import mlflow.pytorch
from GenerativeModels.generative.networks.schedulers import PNDMScheduler, DDPMScheduler, DDIMScheduler
from label_ldm.label_generator.sizeable_inferer import SizeableInferer
from label_ldm.label_generator.labelgen_utils import pad_latent
import nibabel as nib
from tqdm import tqdm
from data.dataset_utils import create_datatext_source_image_gen
from data.dataset import Spade3DSet
from models_spade.pix2pix_model import Pix2PixModel
import pickle
from omegaconf import OmegaConf, DictConfig
from typing import List, Optional, Sequence, Tuple, Union
from utils.util import preprocess_input

class LabelSampler:

    def __init__(self, save_to: str, vae_uri: str, ldm_uri: str, labels: dict,
                 cond_map: dict, wanted_cond: list, cond_boundaries: list,
                 image_shape: list, n_labels: int, formats = ['.nii.gz'], scheduler_type: str = 'ddim',
                 kwargs_scheduler: dict = {}, scale_factor: int = 1.0,
                 ):

        '''

        :param save_to: path where you want to upload the labels
        :param vae_uri: path to vae mlflow folder ([RUN_NO/[BIG NUMBER ID]/artifacts/[best/final]_model
        :param ldm_uri: path to ldm mlflow folder ([RUN_NO/[BIG NUMBER ID]/artifacts/[best/final]_model
        :param labels: Channels of the labels (key: channel position, value: name)
        :param cond_map: In the conditioning, position and conditioning semantics
        :param wanted_cond:  among the conditioning items in cond_map, which (names, as strings) you actually want to sample
        :param cond_boundaries: dictionary containing, for each conditioning token, a list or tuple of
        two elements that are the minimum and maximum bounds of the np.random.uniform that'll be used to sample
        them.
        :param image_shape: list of ints, spatial shape of the samples, must match the one used to train
        the VAE.
        :param n_labels: int, number of labels you want to sample.
        :param formats: list containing the extension (i.e. : '.nii.gz', 'npy') or extenions that you want to save
        the labels under
        :param scheduler_type: str, can only be ddpm ddim or pndm.
        :param kwargs_scheduler: dictionary containing parameters to initialise the schedulers,
         or string leading to the config file used to train the LDM. At least:
        num_train_timesteps that was used, beta_schedule, beta_start and beta_end. Can have
        num_inference_steps if it's DDIM or PNDM.
        :param scale_factor: float, factor used to divide the latent space before going to LDM.
        '''

        self.save_to = save_to
        self.vae_uri = vae_uri
        self.ldm_uri = ldm_uri
        self.list_labels = dict(labels)
        self.cond_map = dict(cond_map)
        self.wanted_cond = list(wanted_cond)
        self.cond_boundaries = dict(cond_boundaries)
        for key, val in self.cond_boundaries.items():
            self.cond_boundaries[key] = list(val)
        self.n_labels= n_labels
        self.formats = list(formats)

        if not os.path.isdir(save_to):
            os.makedirs(save_to)
        for f in self.formats:
            if not os.path.isdir(os.path.join(save_to, "labels_%s" %f.replace(".", ""))):
                os.makedirs(os.path.join(save_to, "labels_%s" %f.replace(".", "")))
        self.json_path = os.path.join(self.save_to, 'label_generation_details.json')
        self.proportions = {}
        for i in wanted_cond:
            self.proportions[i] = 0

        # Load VAE and LDM models
        self.vae = mlflow.pytorch.load_model(self.vae_uri).eval()
        self.ldm = mlflow.pytorch.load_model(self.ldm_uri).eval()
        self.device = torch.device("cuda")
        self.vae = self.vae.to(device=self.device)
        self.ldm = self.ldm.to(device=self.device)
        if type(kwargs_scheduler) in [dict, DictConfig]:
            self.kwargs_scheduler = kwargs_scheduler
        else:
            self.kwargs_scheduler =  dict(OmegaConf.load(kwargs_scheduler)['ldm']['params']['ddpm'])
            if "num_inference_steps" not in self.kwargs_scheduler.keys():
                self.kwargs_scheduler['num_inference_steps'] = 300
        self.scheduler_type = scheduler_type
        # The scheduler type can change (overwritten from the json)
        if scheduler_type.lower() not in ['pndm', 'ddim' , 'ddpm']:
            ValueError("Scheduler type %s not recognised. Must be 'pndm', 'ddim', or 'ddpm'")
        else:
            if scheduler_type == 'pndm':
                self.scheduler = PNDMScheduler(num_train_timesteps=self.kwargs_scheduler['num_train_timesteps'],
                                               beta_schedule=self.kwargs_scheduler['beta_schedule'],
                                               beta_start =self.kwargs_scheduler['beta_start'],
                                               beta_end = self.kwargs_scheduler['beta_end'])
                self.scheduler.set_timesteps(num_inference_steps=self.kwargs_scheduler['num_inference_steps'])
            elif scheduler_type == 'ddim':
                self.scheduler = DDIMScheduler(num_train_timesteps=self.kwargs_scheduler['num_train_timesteps'],
                                               beta_schedule=self.kwargs_scheduler['beta_schedule'],
                                               beta_start =self.kwargs_scheduler['beta_start'],
                                               beta_end = self.kwargs_scheduler['beta_end'])
                self.scheduler.set_timesteps(num_inference_steps=self.kwargs_scheduler['num_inference_steps'])
            else:
                self.scheduler = DDPMScheduler(num_train_timesteps=self.kwargs_scheduler['num_train_timesteps'],
                                               beta_schedule=self.kwargs_scheduler['beta_schedule'],
                                               beta_start =self.kwargs_scheduler['beta_start'],
                                               beta_end = self.kwargs_scheduler['beta_end'],
                                               )

        self.z_shape_vae = [self.vae.latent_channels] + [s // (2 ** (len(self.vae.decoder.ch_mult) - 1)) for s in
                       image_shape]
        need_adjust, z_shape_ldm = pad_latent(self.z_shape_vae, len(self.ldm.block_out_channels))
        self.z_shape_ldm = z_shape_ldm
        self.inferer = SizeableInferer(scheduler=self.scheduler, scale_factor=scale_factor,
                                       latent_shape_vae=self.z_shape_vae, latent_shape_ldm=
                                       self.z_shape_ldm)

        self.scheduler_type = scheduler_type
        self.manage_json(write=True)


    def manage_json(self, write = False, receive = True):
        if write and receive:
            ValueError("Only write or receive can be set to True.")
        config_dict_local = {
           'save_to': self.save_to,
           'vae_uri': self.vae_uri,
           'ldm_uri': self.ldm_uri,
           'list_labels': self.list_labels,
           'cond_map': self.cond_map,
           'wanted_cond': self.wanted_cond,
           'cond_boundaries': self.cond_boundaries,
           'n_labels': self.n_labels,
           'formats': self.formats,
           'proportions': self.proportions,
           'scheduler_type': self.scheduler_type,
            'kwargs_scheduler': self.kwargs_scheduler,
            'z_shape_vae': self.z_shape_vae,
            'z_shape_ldm': self.z_shape_ldm,
        }

        if os.path.isfile(self.json_path):
            with open(self.json_path, 'r') as f:
                config_dict_remote = json.load(f)

        if write:
            f = open(self.json_path, 'w')
            json.dump(config_dict_local,f, indent=4)
        else:
            self.save_to = config_dict_remote['save_to']
            self.vae_uri = config_dict_remote['vae_uri']
            self.ldm_uri = config_dict_remote['ldm_uri']
            self.list_labels = config_dict_remote['list_labels']
            self.cond_map = config_dict_remote['cond_map']
            self.wanted_cond = config_dict_remote['wanted_cond']
            self.cond_boundaries = config_dict_remote['cond_boundaries']
            self.n_labels = config_dict_remote['n_labels']
            self.formats = config_dict_remote['formats']
            self.kwargs_scheduler = config_dict_remote['kwargs_scheduler']
            self.scheduler_type = config_dict_remote['scheduler_type']
            self.z_shape_vae = config_dict_remote['z_shape_vae']
            self.z_shape_ldm = config_dict_remote['z_shape_ldm']

    def get_max_subnbr(self):

        if len(os.listdir(os.path.join(self.save_to, "labels_%s" %self.formats[0].replace(".", "")))) > 0:
            all_subjects  = [int([j for j in i.split("_") if "sub" in j][0].replace("sub-", ""))
                             for i in os.listdir(os.path.join(self.save_to, "labels_%s" %self.formats[0].replace(".", "")))]

            return max(all_subjects)
        else:
            return 0

    def sampleLabel(self, batch_size = 4):

        counter_labels = len(os.listdir(os.path.join(self.save_to, "labels_%s" %self.formats[0].replace(".", ""))))
        n_samples = int(np.ceil((self.n_labels - counter_labels) / batch_size))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        subject_nbr = self.get_max_subnbr()
        if not os.path.isdir(os.path.join(self.save_to, 'examples')):
            os.makedirs(os.path.join(self.save_to, 'examples'))
            n_examples = 0
        else:
            n_examples = len(os.listdir(os.path.join(self.save_to, 'examples')))
        for n_s in range(n_samples):
            cond_list = []
            for c_i, c_name in self.cond_map.items():
                if c_name in self.wanted_cond:
                    cond_list.append(np.random.uniform(self.cond_boundaries[c_name][0],
                                                       self.cond_boundaries[c_name][1],
                                                       batch_size))
                else:
                    cond_list.append(np.asarray([0.0] * batch_size))
            cond = torch.from_numpy(np.stack(cond_list, -1)).unsqueeze(1).type(torch.float).to(device)
            noise = torch.randn([batch_size] + self.z_shape_ldm).to(device)
            samples = self.inferer.sample(
                input_noise=noise,
                autoencoder_model=self.vae,
                diffusion_model=self.ldm,
                scheduler=self.scheduler,
                conditioning=cond,
                save_intermediates=False,
                device=device)
            samples = torch.softmax(samples, 1).detach().cpu().numpy()

            # We verify that we have the lesions we wanted
            argmaxed_samples = np.argmax(samples, 1)

            for b in range(samples.shape[0]):
                if counter_labels >= self.n_labels:
                    break
                lesions_here = []
                sample_b = samples[b, ...]
                for l_ind, l_name in self.list_labels.items():
                    if l_name in self.wanted_cond:
                        if l_ind in argmaxed_samples[b, ...]:
                            lesions_here.append(l_name)
                            self.proportions[l_name] += 1
                if len(lesions_here) == 0:
                    lesions_here = "nolesion"
                else:
                    lesions_here = "-".join(lesions_here)


                # Save
                subject_nbr_ = "0" * (7 - len(str(subject_nbr))) + str(subject_nbr)
                name_file = "Parcellation_sub-%s_%s" %(subject_nbr_, lesions_here)
                sample_b = np.transpose(sample_b, [1, 2, 3, 0])
                for format in self.formats:
                    if format[0] != ".":
                        final_format = ".%s" %format
                    else:
                        final_format = format
                    if "nii.gz" in format:
                        img_ni = nib.Nifti1Image(sample_b, affine = np.eye(4))
                        nib.save(img_ni, os.path.join(self.save_to,
                                                      "labels_%s" %format.replace(".", ""),
                                                      "%s%s" %(name_file, final_format)))
                    elif "npy" in format:
                        np.save(os.path.join(self.save_to,
                                             "labels_%s" %format.replace(".", ""),
                                             "%s%s" %(name_file, final_format)),
                                sample_b)
                    elif "npz" in format:
                        np.savez(os.path.join(self.save_to,
                                             "labels_%s" %format.replace(".", ""),
                                             "%s%s" %(name_file, final_format)),
                                 label = sample_b,
                                 label_affine = np.eye(4))
                    else:
                        ValueError("Unrecognised / unsupported format %s" %format)

                if n_examples < 20:
                    nib_argmax = nib.Nifti1Image(argmaxed_samples[b, ...].astype(float), np.eye(4))
                    nib.save(nib_argmax, os.path.join(self.save_to, 'examples', name_file+".nii.gz"))
                    n_examples += 1
                counter_labels += 1
                subject_nbr += 1

        for key, val in self.proportions.items():
            self.proportions[key] = 100 * (val / self.n_labels)
        self.manage_json(write=True)
        self.createDicts()

    def createDicts(self):

        file_out = os.path.join(self.save_to, "label_dict.tsv")
        create_datatext_source_image_gen(in_dirs=[{'label': os.path.join(self.save_to,
                                                                        "labels_niigz")}],
                                         out_file=file_out, add_dataset=False, new_source=None,
                                         overwrite=True, )



class ImageSampler:

    def __init__(self, save_to: str, data_dict: str, style_dict: str, sequences: list,
                 labels: dict, checkpoints_path: str, formats: list, append_ims: str = "",
                 n_passes: int = 1):

        self.data_dict = data_dict
        self.style_dict = style_dict
        self.labels = dict(labels)
        self.sequences = sequences
        self.checkpoints_path = checkpoints_path
        opt = self.load_overwrite_options()
        self.opt = opt
        self.dataset = Spade3DSet(opt)
        self.save_to = save_to
        self.formats = formats
        self.n_passes = n_passes
        if append_ims != "" and append_ims[-1] != "_":
            self.append_ims = append_ims + "_"
        else:
            self.append_ims = append_ims
        if not os.path.isdir(self.save_to):
            os.makedirs(self.save_to)
        for s in sequences:
            for f in self.formats:
                if not os.path.isdir(os.path.join(self.save_to,"images_%s%s_%s" %(append_ims, s, f.replace(".", "")))):
                   os.makedirs(os.path.join(self.save_to,  "images_%s%s_%s" %(append_ims, s, f.replace(".", ""))))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Pix2PixModel(opt, self.device)
        self.path_json = os.path.join(self.save_to, 'image_generation_details.json')

    def create_config_json(self):

        config_file = {'save_to': self.save_to,
                       'data_dict': self.data_dict,
                       'style_dict': self.style_dict,
                       'sequences': self.sequences,
                       'formats': self.formats,
                       'n_passes': self.n_passes,
                       'append_ims': self.append_ims}

        json.dump(config_file, self.path_json)

    def load_overwrite_options(self, batch_size = 8):

        opt = pickle.load(open(os.path.join(self.checkpoints_path,
                                                      'opt.pkl'), 'rb'))
        opt.data_dict = self.data_dict
        opt.style_dict= self.style_dict
        opt.isTrain = False
        opt.batch_size = batch_size
        opt.non_corresponding_dirs = True
        opt.non_corresponding_ims = False
        opt.perc_val = 0.0
        opt.mode = 'test'
        opt.checkpoints_dir = "/".join(self.checkpoints_path.split("/")[:-1])
        opt.use_ddp = False
        opt.use_dp = False

        return opt

    def sampleImage(self, batch_size =8):

        for seq in self.sequences:
            test_set = self.dataset.resetSeqs(seq)
            loader = monai.data.dataloader.DataLoader(test_set,
                                                      batch_size=batch_size, num_workers=2,
                                                      shuffle=False, drop_last=False)
            for pass_ in range(self.n_passes):
                for lab_batch in tqdm(loader, desc="Seq %s; pass %d/%d" %(seq, pass_, self.n_passes)):
                    input_label, input_image, input_style, modalities = preprocess_input(lab_batch,
                                                                                         device=self.device,
                                                                                         name_mod2num=self.model.name_mod2num,
                                                                                         label_nc=self.opt.label_nc,
                                                                                         contain_dontcare_label=self.opt.contain_dontcare_label)
                    generated = self.model(input_label, input_image, input_style, modalities, mode = 'inference')
                    generated = generated.detach().cpu().squeeze(1)
                    filenames = lab_batch['style_image_meta_dict']['filename_or_obj']
                    for b in range(generated.shape[0]):
                        # We fetch the dataset of the used style
                        dataset = "UNKNOWN"
                        for d in self.dataset.datasets:
                            if d in filenames[b].split("/")[-1]:
                                dataset = d
                        new_name = lab_batch['label_meta_dict']['filename_or_obj'][b].split("/")[-1].split(".")[0].replace("Parcellation", seq)
                        if self.n_passes > 1:
                            new_name = "%s_%s_%d" %(new_name, dataset, pass_)
                        for f in self.formats:
                            if "nii.gz" in f:
                                img_ni = nib.Nifti1Image(generated[b, ...].numpy(), affine = np.eye(4))
                                nib.save(img_ni, os.path.join(self.save_to,
                                                              "images_%s%s_%s" %(self.append_ims,
                                                                                  seq,
                                                                                  f.replace(".", "")),
                                                              new_name+".nii.gz"))
                            elif "npy" in f:
                                np.save(generated[b, ...], os.path.join(self.save_to,
                                                                        "images_%s%s_%s" %(self.append_ims,
                                                                                                  seq,
                                                                                                  f.replace(".", "")),
                                                                        new_name+".npy"),
                                        generated[b, ...].numpy())
                            elif "npz" in f:
                                np.savez(generated[b, ...], os.path.join(self.save_to,
                                                                        "images_%s%s_%s" % (self.append_ims,
                                                                                                seq,
                                                                                                f.replace(".", "")),
                                                                         new_name+".npz"),
                                         img = generated[b, ...].numpy(),
                                         dataset = dataset,
                                         img_affine = np.eye(4))