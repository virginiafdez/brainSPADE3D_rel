import os
import torch
import torch.nn as nn
import monai

# Neural Network class
class Modisc(nn.Module):
    def __init__(self, n_mods, n_datasets, dropout, spatial_dims, in_channels):
        '''
        Modality Classifier with Densenet-2 branches.
        :param n_mods: number of modalities to classify
        :param n_datasets: number of datasets to classify
        :param dropout: proportion of dropout (0-1)
        :param spatial_dims: number of spatial dimensions (1D, 2D, 3D etc.)
        :param in_channels: in channels (RGB 3, grayscale 1 etc).
        '''
        super(Modisc, self).__init__()
        self.base_model = monai.networks.nets.densenet121(spatial_dims=spatial_dims, in_channels=in_channels,
                                                              out_channels=n_mods,
                                                              dropout_prob=dropout)
        self.ds_layer_1 = nn.Sequential(nn.ReLU(),
                               nn.AdaptiveAvgPool2d(output_size=1))
        self.ds_layer_2 = nn.Linear(in_features=1024, out_features=n_datasets, bias = True)

    def forward(self, x):
        '''
        Passes a batch through the main body of the Densenet, then separately
        returns the modality classification, and the dataset classification.
        :param x:
        :return:
        '''
        x_out = self.base_model.features(x)
        x_mod = self.base_model.class_layers(x_out)
        x_dat = self.ds_layer_1(x_out)
        x_dat = x_dat.view(x_dat.shape[0],-1)
        x_dat = self.ds_layer_2(x_dat)
        return x_mod, x_dat

    def findLastModel(self, path_to_model):
        '''
        Loads previous instance of hte model.
        Looks for pth files wih EPXXX where XXX = Epoch number
        Only loads if epoch > 1
        :param path_to_model: Path where the pth nets are stored.
        :return:
        '''
        all_files = os.listdir(os.path.join(path_to_model))
        all_files = [a for a in all_files if ".pth" in a]
        epoch_last = 1
        file_last = None
        for f in all_files:
            epoch = int(f.split("EP")[-1].split(".pth")[0])
            if epoch_last < epoch:
                epoch_last = epoch
                file_last = f
        if epoch_last > 1:
            self.load_state_dict(torch.load(os.path.join(path_to_model, file_last)), strict=False)
            print("Loading state_dict from epoch %d" %epoch_last)
        else:
            print("Could not find / load state_dict in %s" %path_to_model)
