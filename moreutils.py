import numpy as np
#import torchvision.utils
import os
import torch
# For training on cluster, uncomment these 2 lines
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from copy import deepcopy
from utils import util
from scipy.ndimage import gaussian_filter
from matplotlib.pyplot import boxplot as boxplot
import statsmodels.api as sm
import nibabel as nib
import csv
import scipy
from collections import Counter
import numpy as np
from scipy.ndimage import label, binary_dilation
from PIL import Image
import torchvision
import monai

def saveUncert(img, save_path):

    img = img.detach().cpu() # We convert the image to numpy (we don't need it anymore)
    max_img = img.max()
    min_img = img.min()
    img = (img-min_img)/(max_img-min_img) # Normalize
    img = img.unsqueeze(0)
    torchvision.utils.save_image(img, save_path, normalize = True) # Just in case!

def saveFigs(imgs, fig_path, create_dir = False, nlabels = 1,
             sequences = [], suptitle = ""):

    '''
    Plots a series of images next to another for one element of the batch or all the batch.
    :param imgs: Dictionary containing images
    :param fig_path:Path + name to save the file
    :param create_dir: Whether the directory to save the images must be created or not.
    :param nlabels: Number of labels in the label image.
    :param index: If -1, all the batches will be plotted in a different subplot row. Otherwise, index
    is the position (0 to B-1) of the image within a specific batch that will be plotted
    :return:
    '''

    # If directory isn't created, we create it.
    if create_dir:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    # If index has a value different from -1, the index-th element of the images lists will be plotted.
    n_rows = imgs[list(imgs.keys())[0]].shape[0]  # Batch size
    # Number of subplots
    n_subplots = n_rows

    # Before plotting the figure, we make duplicates, post-processing the images.
    maxes = -10000
    mins = 10000
    imgs_to_plot = {}
    for key, value in imgs.items():
        if 'label' in key:
            temp_i = util.tensor2label(value, nlabels)
            if len(temp_i.shape) == 5:
                temp_i, _ = chopSlice(temp_i[..., 0], cut='a', mode = 'middle')
                imgs_to_plot[key] = temp_i
            else:
                imgs_to_plot[key] = temp_i[..., 0]
        else:
            if len(value.shape) == 5:
                temp_i, _ = chopSlice(util.tensor2im(value)[..., 0], cut = 'a', mode='middle')
                temp_i = np.stack([temp_i]*3, -1)
            else:
                temp_i = util.tensor2im(value, is2d = True)
            imgs_to_plot[key] = temp_i
        mins_ = min([i.min() for i in imgs_to_plot[key]])
        maxs_ = max([i.max() for i in imgs_to_plot[key]])
        if mins_ < mins:
            mins = mins_
        if maxs_ > maxes:
            maxes = maxs_

    # Plot figures.
    f = plt.figure(figsize = (7, 2*n_rows))
    subplot_counter = 1
    for b in range(n_rows):
        title_ = ", ".join(imgs_to_plot.keys())
        if len(sequences) != 0:
            title_ = "%s (%s)" %(title_, sequences[b])
        to_plot_array = []
        for item, value in imgs_to_plot.items():
            if 'label' in item:
                to_plot_array.append(get_rgb_colours()[value])
            else:
                to_plot_array.append(value)

        # We need to make sure the images we can concatenate
        lista_maxes = np.asarray([i.shape[1:] for i in to_plot_array]).max(0)
        padder = monai.transforms.SpatialPad(lista_maxes, mode="constant", value=mins)
        for ind, itemim in enumerate(to_plot_array):
            to_plot_array[ind] = padder(np.expand_dims(itemim[b, ...], 0))[0, ...]

        plt.subplot(n_rows, 1, subplot_counter)
        plt.imshow(np.concatenate(to_plot_array, 1), vmin=mins, vmax=maxes)
        plt.axis('off')
        plt.title(title_)

        subplot_counter+=1

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(f)


def saveFigs_stacks(imgs, fig_path, create_dir=False, nlabels=1, index=-1, same_scale=False, titles=[],
             batch_accollades={}, index_label=-1 , bound_normalization = False):
    '''
    Plots a series of images next to another for one element of the batch or all the batch.
    Plots images (not label) stacked together.
    :param imgs: Tuple or list of N Bx1/3xHxW tensors or numpy images.
    :param fig_path:Path + name to save the file
    :param titles: Names to give to each of the N different types of images provided. Must be a list
    of the same size as N (not B!). The titles will be the same for each batch.
    :param create_dir: Whether the directory to save the images must be created or not.
    :param nlabels: Number of labels in the label image.
    :param index: If -1, all the batches will be plotted in a different subplot row. Otherwise, index
    is the position (0 to B-1) of the image within a specific batch that will be plotted
    :param same_scale: Whether you want to plot all (N-1) images within a row/batch, excluding the label,
    in the same min-max scale
    :param batch_accollades: dictionary of key-string (in case that index is not (-1)) or key-list (where
    the list is of size B) of things that must be changed from one batch to another in the titles. Example:
    key: sequence, value = [T1, FLAIR, T1, FLAIR], titles = [Input (sequence), ...]. For each item of the
    batch, the title would be Input (T1), Input (FLAIR) etc.
    :param index_label: the first element of args is the index within N (from 0 to N-1) in which the label/
    segmentation map is.
    :param bound_normalization: boolean, if True, images are assumed to be bounded between -1 and 1.
    :return:
    '''

    if create_dir:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    if index == -1:
        subplots = True
    else:
        subplots = False

    n_columns = len(imgs)

    for ind_img, img in enumerate(imgs):
        if len(img.shape) < 4:
            if torch.is_tensor(img):
                imgs[ind_img] = img.unsqueeze(0)
            subplots = False
            index = 0

    if subplots:
        # The number of batches is the first dimension.
        # If dim() is 3, there is no batch (there's a single image), so we skip
        # and set the number of rows to 1.
        n_rows = imgs[0].shape[0]  # N_Batches
        batch_size = imgs[0].shape[0]
        relevant_indices = list(range(0, batch_size))
    else:
        n_rows = 1
        batch_size = imgs[0].shape[0]
        relevant_indices = [index]

    n_subplots = 1 * n_rows

    if titles == []:
        titles = ["Image"] * n_columns

    # Batch acollades is a key-value pair where the key is a placeholder on the title
    # and the value is a list containing the value for which the placeholder will be replaced
    # within each batch.

    for key, value in batch_accollades.items():
        for title in titles:
            if key in title:
                if subplots:
                    if len(value) != n_rows:
                        ValueError("The number of batch acollades for %s must match the batch size. Expected %d"
                                   "but found %d." % (key, n_rows, len(value)))

    f = plt.figure(figsize=(7 * n_columns, 5 * n_rows))

    subplot_counter = 1
    for b in range(batch_size):
        if b not in relevant_indices:
            continue
        # Initialisation
        b_imgs = []
        maxes = -10000
        mins = 10000
        # Titles
        titles_ = []
        for title in titles:
            for key, value in batch_accollades.items():
                if key in title:
                    if subplots:
                        titles_.append(title.replace(key, value[b]))
                    else:
                        titles_.append(title.replace(key, value))
                else:
                    titles_.append(title)
        if titles_ == []:
            titles_ = titles

        for ind_i, i in enumerate(imgs):

            if ind_i == index_label:
                if torch.is_tensor(i[b:b + 1, ...]):
                    i_ = util.tensor2label(i[b:b + 1, ...], nlabels)
                else:
                    i_ = i
                i_ = handleSizes(i_, rgb_ok=True)
                b_imgs.append(i_)
            else:
                if torch.is_tensor(i[b:b + 1, ...]):
                    i_ = util.tensor2im(i[b:b + 1, ...], bound_normalization = bound_normalization)
                else:
                    i_ = i
                i_ = handleSizes(i_)
                if i_.min() < mins:
                    mins = i_.min()
                if i_.max() > maxes:
                    maxes = i_.max()
                b_imgs.append(np.stack([i_]*3, -1))

        b_imgs = np.concatenate(b_imgs, 1)
        plt.subplot(n_rows, 1, subplot_counter)
        plt.title(", ".join(titles_), fontsize = 18)
        plt.imshow(b_imgs)
        plt.axis('off')
        subplot_counter += 1

    plt.savefig(fig_path)
    plt.close(f)

def saveFigs_1row(real, synt, fig_path, create_dir = False, sequence = "", divide = 1,
                  bound_normalization = False):
    '''
    Plots the label, input style figure and synthetised image concatenation in a row.
    :param label: tensor, label
    :param real: tensor, image
    :param synt:  synthetic image
    :param fig_path: figure where you want to save this
    :param create_dir: do you want to create the directory if it doesn't exist?
    :param nlabels: number of labels
    :param sequence: which style is it
    :param same_scale: plots real and synt in the same contrast.
    :param bound_normalization: if True, images are assumed to be bounded between -1 and 1.
    :return:
    '''

    # Create directory if needed
    if create_dir:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    if torch.is_tensor(real):
        real = util.tensor2im(real, normalize= False, bound_normalization = bound_normalization)

    # For the synthetic images: they should be N x C x H x W
    # We want: H x (NxW)

    column_index = 1 # Number of images we've got per row
    row_index = 1
    total_per_row = int((synt.shape[0] + 1) / divide)
    first_row = real
    next = None

    for sample_id in range(synt.shape[0]):
        sample = synt[sample_id, ...]
        if torch.is_tensor(sample):
            sample[sample<0] = 0.0
            sample = util.tensor2im(sample, normalize=False, bound_normalization = bound_normalization)
            sample = handleSizes(sample)

        column_index += 1

        if column_index > total_per_row:
            # If we are exceeding the number of images per row
            if row_index == 1:
                OUT = first_row
            else:
                OUT = np.concatenate((OUT, next), axis = 0) # On height channel.
            row_index += 1
            column_index = 1
            next = sample
        else:
            # Don't need to change rows, we enchain columns
            if row_index == 1:
                first_row = np.concatenate((first_row, sample), axis=1)  # On width channel.
            else:
                next = np.concatenate((next, sample), axis=1)  # On width channel.

    if next is not None:
        try:
            OUT = np.concatenate((OUT, next), axis=0)
        except:
            Warning("Omitted last row on the plot because the 'divide' arg is not a multiple of the"
                    "generated + 1 number. Consider modifying the parameter")

    # We do imshow
    plt.figure(figsize=(10, 10))
    plt.imshow(OUT, cmap = 'gist_gray')
    plt.title("Input style (%s). First: input." % sequence)
    plt.colorbar(fraction=0.03)
    plt.savefig(fig_path)
    plt.close()
def plotHeatmaps(original, tweaked, codes, save_name, bound_normalization = False):

    n_codes = tweaked.shape[0]
    n_subplots = int(np.ceil(np.sqrt(n_codes + 1)))
    original = util.tensor2im(original, normalize=True, bound_normalization = bound_normalization)
    f = plt.figure(figsize=(n_subplots*6, n_subplots*4))
    plt.subplot(n_subplots, n_subplots, 1)
    plt.imshow(original)
    subplot_index = 2
    for c in range(n_codes):
        plt.subplot(n_subplots, n_subplots, subplot_index)
        code_im = util.tensor2im(tweaked[c, ...], bound_normalization = bound_normalization)
        difference = (code_im - original)/original
        difference[np.isnan(difference)] = 0.0
        difference = np.round(difference*255).astype('uint8')[:,:,1]
        plt.subplot(n_subplots, n_subplots, subplot_index)
        plt.imshow(code_im)
        plt.imshow(difference, alpha = 0.4, cmap = plt.get_cmap('jet'))
        plt.colorbar(fraction=0.035)
        plt.title("Code %d" %(codes[c]))
        subplot_index +=1
    plt.savefig(save_name)
    plt.close(f)
def plotMus(original, mu, save_name, bound_normalization = False):

    original = util.tensor2im(original, normalize=True, bound_normalization = bound_normalization)
    mu = util.tensor2im(mu, normalize=True, bound_normalization = bound_normalization)
    f = plt.figure(figsize=(3*6, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.imshow(mu)
    plt.title("Mean")
    plt.subplot(1, 3, 3)
    difference = ((mu - original)/original)
    difference[np.isnan(difference)] = 0.0
    difference = np.round(difference * 255).astype('uint8')[:, :, 1]
    plt.imshow(difference, alpha=0.4, cmap=plt.get_cmap('jet'))
    plt.colorbar(fraction=0.035)
    plt.title("Difference")

    plt.savefig(save_name)
    plt.close(f)
def handleSizes(img, rgb_ok = False):
    if isinstance(img, np.ndarray):
        i = deepcopy(img)
        if i.ndim > 3:
            i = i[0, :,:,:]
        # Channel dim can be in position 2 or 1
        if i.shape[0] <= 3:
            channels = i.shape[0]
            if channels != 3:
                i = i[0, :, :]
            elif channels == 3:
                if not rgb_ok:
                    i = i[1, :, :]  # We keep the central channel
                else:
                    i = np.transpose(i, [1, 2, 0])
        elif i.shape[2] <= 3:
            channels = i.shape[2]
            if channels != 3:
                i = i[:, :, 0]
            if channels == 3:
                if not rgb_ok:
                    i = i[:, :, 1]  # We keep the central channel
                else:
                    pass
    elif torch.is_tensor(img):
        i = img.clone()
        if i.dim() > 3:
            if i.shape[0] > 1:
                i = i[0, -1]
            else:
                i = i.squeeze(0)

        # Channel dim can be in position 2 or 1
        if i.shape[0] <= 3:
            channels = i.shape[0]
            if channels != 3:
                i = i[0, :, :]
            elif channels == 3:
                if not rgb_ok:
                    i = i[1, :, :]  # We keep the central channel
                else:
                    i = i.permute(1, 2, 0)
        elif i.shape[2] <= 3:
            channels = i.shape[2]
            if channels != 3:
                i = i[:, :, 0]
            if channels == 3:
                if not rgb_ok:
                    i = i[:, :, 1]  # We keep the central channel
                else:
                    pass
    else:
        TypeError("Unknown type.")

    return i
def pair_up(list_labs):
    np.random.shuffle(list_labs)
    pairs = [];
    if len(list_labs) % 2 != 0:
        cleeve = int((len(list_labs)-1)/2)
        pairs = zip(list_labs[1:cleeve], list_labs[cleeve:-1])
    else:
        cleeve = int((len(list_labs))/2)
        pairs = list(zip(list_labs[1:cleeve], list_labs[cleeve:]))
    return pairs
def structural_Similarity(img1, img2, mean = False):
    '''
    Calculates the Structural Similarity Index (SSIM) between two tensors
    :param img1, tensor 1
    :param img2, tensor 2
    :return ssim index (between 0 and 1)
    '''
    out = []

    img1_c = img1.clone()
    img2_c = img2.clone()
    if img1.dim() != img2.dim():
        ValueError("Dimensions must match!")

    for b in range(img1_c.shape[0]):
        sub_img1_c = img1_c[b,...]
        sub_img2_c = img2_c[b,...]
        size1 = sub_img1_c.shape
        size2 = sub_img2_c.shape
        if size1[0] == 1:
            sub_img1_c = sub_img1_c.squeeze(0) # Again, we shrink the channel dimension
            sub_img1_c = sub_img1_c.detach().cpu().numpy()
        elif size1[0] == 3:
            sub_img1_c = sub_img1_c[1, :, :] # Again, we shrink the channel dimension
            sub_img1_c = sub_img1_c.detach().cpu().numpy()
        elif size1[2] ==1:
            sub_img1_c = sub_img1_c.squeeze(2) # Again, we shrink the channel dimension
            sub_img1_c = sub_img1_c.detach().cpu().numpy()
        elif size1[2] == 3:
            sub_img1_c = sub_img1_c[:, :, 1] # Again, we shrink the channel dimension
            sub_img1_c = sub_img1_c.detach().cpu().numpy()
        else:
            ValueError("Unknown dimension for im1. Either 0th or 2d dims must be 1 or 3.")

        if size2[0] == 1:
            sub_img2_c = sub_img2_c.squeeze(0) # Again, we shrink the channel dimension
            sub_img2_c = sub_img2_c.detach().cpu().numpy()
        elif size2[0] == 3:
            sub_img2_c = sub_img2_c[1, :, :] # Again, we shrink the channel dimension
            sub_img2_c = sub_img2_c.detach().cpu().numpy()
        elif size2[2] ==1:
            sub_img2_c = sub_img2_c.squeeze(2) # Again, we shrink the channel dimension
            sub_img2_c = sub_img2_c.detach().cpu().numpy()
        elif size2[2] == 3:
            sub_img2_c = sub_img2_c[:, :, 1] # Again, we shrink the channel dimension
            sub_img2_c = sub_img2_c.detach().cpu().numpy()
        else:
            ValueError("Unknown dimension for im2. Either 0th or 2d dims must be 1 or 3.")

        if img1_c.shape[0] == 1:
            if torch.is_tensor(sub_img1_c):
                sub_img1_c = sub_img1_c.numpy()
            if torch.is_tensor(sub_img2_c):
                sub_img2_c = sub_img2_c.numpy()
            out = [ssim(sub_img1_c, sub_img2_c)]
        else:
            if torch.is_tensor(sub_img1_c):
                sub_img1_c = sub_img1_c.numpy()
            if torch.is_tensor(sub_img2_c):
                sub_img2_c = sub_img2_c.numpy()
            out.append(ssim(sub_img1_c, sub_img2_c))
    if mean:
        return np.mean(out)
    return out

def l2(im1, im2):
    '''
    Returns the L2 norm between tensors.
    :param im1: Image 1 (tensor, B x C x H x W)
    :param im2: (tensor, B x C x H x W)
    :return:
    '''
    if im1.shape[0] > 1:
        out = []
        for b in range(im1.shape[0]):
            out.append(torch.sqrt((im1[b,...]-im2[b,...]).pow(2).sum()))
        return out
    else:
        return [float(torch.sqrt((im1-im2).pow(2).sum()))]

def normalizedMutualInfo(im1, im2, bins = 125):
    """
    Computes the Normalized Mutual Information Between two images
    :param im1: Pytorch tensor image
    :param im2: Pytorch tensor image
    :return: NMI
    """
    im1_bis = im1.clone()
    im2_bis = im2.clone()
    # We go from tensor to numpy
    im1_bis = im1_bis.detach().cpu().numpy()
    im2_bis = im2_bis.detach().cpu().numpy()
    # We compute the 2D histogram
    joint_his, _, _ = np.histogram2d(im1_bis.ravel(), im2_bis.ravel(), bins)

    # NMI
    # Need to fetch probabilities
    joint_prob = joint_his / np.sum(joint_his)  # Normalize histogram
    prob1 = np.sum(joint_prob, axis=1)
    prob2 = np.sum(joint_prob, axis=0)
    prob1_prob2 = prob1[:, None] * prob2[None, :]
    joint_prob_NZ = joint_prob > 0
    # Mutual Information
    MI = np.sum(joint_prob[joint_prob_NZ] * np.log(joint_prob[joint_prob_NZ] / prob1_prob2[joint_prob_NZ]))

    # Entropy
    prob1_NZ = prob1 > 0
    entropy1 = -np.sum(prob1[prob1_NZ] * np.log(prob1[prob1_NZ]))
    prob2_NZ = prob2 > 0
    entropy2 = -np.sum(prob2[prob2_NZ] * np.log(prob2[prob2_NZ]))

    return (2 * MI / (entropy1 + entropy2))
def plotTrainingLosses(loss_file, *args):
    if len(args) > 0:
        savedir = args[0]
    else:
        savedir = loss_file.split('/')
        savedir = "/".join(savedir[:-1])

    losses = {} # All iterations in one epoch
    losses_av = {} # Per epoch
    epoch_vector = []
    iter_vector = []
    modalities = {"T1": 'darkblue', "FLAIR": 'teal'}

    better_names = {'KLD': "KLD Loss", "GAN": "Total GAN Loss", "VGG": "VGG Loss", "GAN_Feat": "Feature Loss (GAN)", "D_Fake": "Discriminator (on fake)", "D_real": "Discriminator (on real)", "coherence": "NMI inter-sequence loss"}
    f = open(loss_file, 'r')
    lines = f.readlines()

    last_epoch = 0

    for ind_line, line in enumerate(lines):
        if '(epoch' in line:
            splitline = line.split('(')
            splitline = splitline[-1]
            splitline = splitline.split(')')
            epoch, iter = process_Epoch_Stream(splitline[0])
            iter_vector.append(epoch*iter)
            if epoch != last_epoch:
                last_epoch = epoch
                epoch_vector.append(epoch)

                # With each change of epoch, we empty the losses
                if epoch != 1: # Because we have no data for "epoch 0"
                    for loss_type, loss_val_dict in losses.items():
                        for loss_mod, loss_vector in loss_val_dict.items():
                            if loss_type not in losses_av.keys():
                                losses_av[loss_type] = {}
                                losses_av[loss_type][loss_mod] = [np.mean(loss_vector)]
                            else:
                                if loss_mod not in losses_av[loss_type].keys():
                                    losses_av[loss_type][loss_mod] = [np.mean(loss_vector)]
                                else:
                                    losses_av[loss_type][loss_mod].append(np.mean(loss_vector))
                    losses = {}

            splitline = splitline[-1] # We keep the losses

            losses_line = splitline.split(";")
            losses_line = losses_line[:-1] # Last item is a linebreak
            for loss in losses_line:
                loss_sp = loss.split(':')
                loss_id = loss_sp[0] # Loss ID
                loss_id = loss_id.replace(" ", "")
                try:
                    if better_names[loss_id] not in losses.keys():
                        losses[better_names[loss_id]] = {} # We try the better name if it's in the dictionnary
                    loss_id = better_names[loss_id]
                except:
                    if loss_id not in losses.keys():
                        losses[loss_id] = {} # Otherwise we put this one

                loss_sp = loss_sp[-1]
                loss_sp = loss_sp.split(" ") # Split on spaces
                if loss_sp[0] == "":
                    del loss_sp[0]

                ranges = np.arange(0, len(loss_sp), 2)
                for r in ranges:
                    if loss_sp[r] not in losses[loss_id].keys():
                        losses[loss_id][loss_sp[r]] = [np.round(float(loss_sp[r+1]),3)] # Append the rth+1 element (value for the modality) with the modality (rth element) as key
                    else:
                        losses[loss_id][loss_sp[r]].append(np.round(float(loss_sp[r+1]), 3))

    # Last epoch processing
    for loss_type, loss_val_dict in losses.items():
        for loss_mod, loss_vector in loss_val_dict.items():
            if loss_type not in losses_av.keys():
                losses_av[loss_type] = {}
                losses_av[loss_type][loss_mod] = [np.mean(loss_vector)]
            else:
                if loss_mod not in losses_av[loss_type].keys():
                    losses_av[loss_type][loss_mod] = [np.mean(loss_vector)]
                else:
                    losses_av[loss_type][loss_mod].append(np.mean(loss_vector))
    losses = {}


    # Discriminators in the same plot!

    f = plt.figure(figsize = (15, 15)) # Plot a figure

    # Number of losses
    n_losses = len(losses_av.keys())-1 # Discriminators go together

    # Number of subplots
    n_plots = int(np.ceil(np.sqrt(n_losses)))

    plot_ID = 1
    this_legend = []

    for key, lossval in losses_av.items():
        plt.subplot(n_plots, n_plots, plot_ID)

        if 'Discriminator' in key:
            continue

        for mod, mod_color in modalities.items():
            plt.plot(epoch_vector, lossval[mod], ':', color = mod_color)
            this_legend.append(mod)

        plt.legend(this_legend)
        plt.title(key)
        plt.xlabel("Iteration")
        plt.ylabel("Loss value")

        plot_ID = plot_ID+1

    # We need to plot discriminator losses
    plt.subplot(n_plots, n_plots, plot_ID)
    this_legend = []
    for mod, mod_color in modalities.items():
        #plt.plot(epoch_vector, losses_av['Discriminator (on fake)'][mod], ':', color=mod_color)
        #plt.plot(epoch_vector, losses_av['Discriminator (on real)'][mod], ':', color=mod_color)
        sum_losses = [i + losses_av['Discriminator (on fake)'][mod][ind] for ind, i in enumerate(losses_av['Discriminator (on real)'][mod])]
        plt.plot(epoch_vector, sum_losses, linestyle='-', color=mod_color)
        #this_legend.append(mod+" (on fake)")
        #this_legend.append(mod+" (on real)")
        this_legend.append(mod + " (TOTAL)")
    plt.legend(this_legend)
    plt.title(key)
    plt.xlabel("Iteration")
    plt.ylabel("Loss value")

    plt.savefig(os.path.join(savedir, 'training_losses.png'))

    plt.close()
def plot_Histogram(realT1, fakeT1, realFLAIR, fakeFLAIR, save_name):

    kl_value = {}

    realT1 = realT1.flatten()
    fakeT1 = fakeT1.flatten()
    realFLAIR = realFLAIR.flatten()
    fakeFLAIR = fakeFLAIR.flatten()
    realT1 = realT1[realT1>=1]
    fakeT1 = fakeT1[fakeT1 >= 1]
    realFLAIR = realFLAIR[realFLAIR >= 1]
    fakeFLAIR = fakeFLAIR[fakeFLAIR >= 1]

    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    plt.hist(realT1.flatten(), alpha = 0.8, color = 'lightgreen', label = "Real", bins = 75)
    plt.hist(fakeT1.flatten(),  alpha = 0.4, color = 'dodgerblue', label = "Fake", bins = 75)
    plt.legend()
    plt.title("T1")

    hist_T1_real, bin_edges = np.histogram(realT1.flatten(), bins = 75)
    hist_T1_fake, _ = np.histogram(fakeT1.flatten(), bins = bin_edges)
    kl_value['T1'] = KL_Divergence(hist_T1_fake, hist_T1_real)

    plt.subplot(1, 2, 2)
    plt.hist(realFLAIR.flatten(),  alpha = 0.8, color = 'lightgreen', label = "Real", bins = 75)
    plt.hist(fakeFLAIR.flatten(),  alpha = 0.4, color = 'dodgerblue', label = "Fake", bins = 75)
    plt.legend()
    plt.title("FLAIR")

    hist_FLAIR_real, bin_edges = np.histogram(realFLAIR.flatten(), bins = 75)
    hist_FLAIR_fake, _ = np.histogram(fakeFLAIR.flatten(), bins = bin_edges)
    kl_value['FLAIR'] = KL_Divergence(hist_FLAIR_fake, hist_FLAIR_real)

    plt.savefig(save_name)

    plt.close()

    return kl_value

def KL_Divergence(p, q, epsilon = 0.000001):
    epsilon = epsilon
    # We want probabilities, so we normalise
    p = np.asarray(p/sum(p), dtype=np.float) + epsilon
    q = np.asarray(q/sum(q), dtype=np.float) + epsilon
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def plotTrainingLosses_1dec(loss_file, smooth = False, *args):
    if len(args) > 0:
        savedir = args[0]
    else:
        savedir = loss_file.split('/')
        savedir = "/".join(savedir[:-1])

    losses = {} # All iterations in one epoch
    losses_av = {} # Per epoch
    epoch_vector = []
    iter_vector = []
    modalities = {"T1": 'darkblue', "FLAIR": 'teal'}
    better_names = {'KLD': "KLD Loss", "GAN": "Total GAN Loss", "VGG": "VGG Loss", "GAN_Feat": "Feature Loss (GAN)", "D_Fake": "Discriminator (on fake)",
                    "D_real": "Discriminator (on real)", "coherence": "NMI inter-sequence loss",
                    "D_acc_fakes": "Discrim. Acc. (Fakes)", "D_acc_reals": "Discrim. Acc. (Reals)",
                    "D_acc_total": "Discrim. Acc.", "mod_disc": "Modality Discrimination"}
    f = open(loss_file, 'r')
    lines = f.readlines()

    last_epoch = 0

    for ind_line, line in enumerate(lines):
        if '(epoch' in line:
            splitline = line.split('(')
            splitline = splitline[-1]
            splitline = splitline.split(')')
            epoch, iter = process_Epoch_Stream(splitline[0])
            iter_vector.append(epoch*iter)
            if epoch != last_epoch:
                # Before continuing processing the line, we dump the previous epoch info into losses_av.
                # losses_av has the mean per epoch.
                last_epoch = epoch
                epoch_vector.append(epoch)

                # With each change of epoch, we empty the losses
                if epoch != 1: # Because we have no data for "epoch 0"
                    # Ie. If epoch is 2, we are dumping info about epoch 1.
                    for loss_type, loss_val in losses.items():
                        if loss_type not in losses_av.keys():
                            losses_av[loss_type] = {}
                            losses_av[loss_type] = [np.mean(loss_val)]
                        else:
                            losses_av[loss_type].append(np.mean(loss_val))
                    losses = {}

            splitline = splitline[-1] # We keep the losses
            losses_line = splitline.split(";")
            losses_line = losses_line[:-1] # Last item is a linebreak
            for loss in losses_line:
                loss_sp = loss.split(':')
                loss_id = loss_sp[0] # Loss ID
                loss_id = loss_id.replace(" ", "")
                loss_sp = loss_sp[-1]
                loss_sp = loss_sp.split(" ")  # Split on spaces
                if loss_sp[0] == "":
                    del loss_sp[0]
                try:
                    if better_names[loss_id] not in losses.keys():
                        losses[better_names[loss_id]] = [] # We try the better name if it's in the dictionnary
                    loss_id = better_names[loss_id]
                    if loss_id not in losses.keys():
                        losses[loss_id] = [] # Otherwise we put this one
                        losses[loss_id] = float(loss_sp[0])  # Sole loss value
                    else:
                        losses[loss_id].append(float(loss_sp[0]))
                except:
                    if loss_id not in losses.keys():
                        losses[loss_id] = [] # Otherwise we put this one
                        losses[loss_id] = [float(loss_sp[0])] # Sole loss value
                    else:
                        losses[loss_id].append(float(loss_sp[0]))


    # Last epoch processing
    for loss_type, loss_val in losses.items():
        if loss_type not in losses_av.keys():
            losses_av[loss_type] = [np.mean(loss_val)]
        else:
            losses_av[loss_type].append(np.mean(loss_val))
    losses = {}

    # Make sure that the losses are leveraged
    max_len = max([len(v) for k, v in losses_av.items()])
    for k, v in losses_av.items():
        n_z = max_len - len(v)
        losses_av[k] = [0]*n_z + v

    # Discriminators in the same plot!
    f = plt.figure(figsize = (30, 20)) # Plot a figure

    # Number of losses
    n_losses = len(losses_av.keys())-1 #-2 # Discriminators go together, accuracies as well

    # Number of subplots
    n_plots = int(np.ceil(np.sqrt(n_losses)))

    plot_ID = 1
    this_legend = []

    for key, lossval in losses_av.items():
        plt.subplot(n_plots, n_plots, plot_ID)

        if 'Discriminator' in key:
            plot_disc = plot_ID
            continue

        if "Acc" in key:
            plot_acc = plot_ID
            continue

        new_epoch_vector = np.arange(0, len(lossval), 1)

        if smooth:
            filtered = sm.nonparametric.lowess(lossval, new_epoch_vector, frac=0.1, return_sorted=False)
            plt.plot(new_epoch_vector, filtered, ':', color='teal')
        else:
            plt.plot(new_epoch_vector, lossval, ':', color = 'teal')

        plt.title(key, fontsize=22)
        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Loss value", fontsize=16)

        plot_ID = plot_ID+1

    # We need to plot discriminator losses
    plt.subplot(n_plots, n_plots, plot_ID)
    new_epoch_vector = np.arange(0, len(losses_av['Discriminator (on fake)']), 1)
    if smooth:
        filtered_1 =  sm.nonparametric.lowess(losses_av['Discriminator (on fake)'], new_epoch_vector, frac=0.1, return_sorted=False)
        filtered_2 =  sm.nonparametric.lowess(losses_av['Discriminator (on real)'], new_epoch_vector, frac=0.1, return_sorted=False)
        plt.plot(new_epoch_vector, filtered_1, ':', color='teal')
        plt.plot(new_epoch_vector, filtered_2, ':', color='firebrick')
    else:
        plt.plot(new_epoch_vector, losses_av['Discriminator (on fake)'], ':', color='teal')
        plt.plot(new_epoch_vector, losses_av['Discriminator (on real)'], ':', color='firebrick')
    plt.title('Discriminator loss', fontsize=22)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)

    plot_ID += 1

    # We need to plot discriminator accuracies
    plt.subplot(n_plots, n_plots, plot_ID)
    new_epoch_vector = np.arange(0, len(losses_av['Discrim. Acc. (Fakes)']), 1)

    if smooth:
        filtered_fakes = sm.nonparametric.lowess(losses_av['Discrim. Acc. (Fakes)'], new_epoch_vector, frac=0.1, return_sorted=False)
        filtered_reals = sm.nonparametric.lowess(losses_av['Discrim. Acc. (Reals)'], new_epoch_vector, frac=0.1, return_sorted=False)
        filtered_totals = sm.nonparametric.lowess(losses_av['Discrim. Acc.'], new_epoch_vector, frac=0.1, return_sorted=False)
        plt.plot(new_epoch_vector, filtered_fakes, color='teal', alpha=0.5)
        plt.plot(new_epoch_vector, filtered_reals, color='firebrick', alpha=0.5)
        plt.plot(new_epoch_vector, filtered_totals, '-', color='grey', linewidth=2)
    else:
        plt.plot(new_epoch_vector, losses_av['Discrim. Acc. (Fakes)'], color='teal', alpha = 0.5)
        plt.plot(new_epoch_vector, losses_av['Discrim. Acc. (Reals)'], color='firebrick', alpha = 0.5)
        plt.plot(new_epoch_vector, losses_av['Discrim. Acc.'], '-', color='grey', linewidth = 2)
    plt.title('Discriminator Accuracy', fontsize=22)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Accuracy value", fontsize=16)
    plt.legend(["On fakes", "On reals", "Total"])

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(os.path.join(savedir, 'training_losses.png'))

def saveSingleImage(img, label, path, skullstrip, denormalize = False,
                    islabel = False, create_dir = False,
                    save_as_grey = False):
    '''
    Saves a single image as a PIL image.
    :param img: Image (in torch, numpy format)
    :param label: Label corresponding to the image (for Skull Strip)
    :param path: Where you want to store it and the name
    :param skullstrip: Whether you want to skull strip
    :param save_as_grey: whether to save as greyscale float
    '''

    if denormalize and not islabel and not save_as_grey:
        img = (img + 1) / 2.0 * 255.0
    if skullstrip and not islabel:
        img = SkullStrip(img, label, value = img.min())
    if img.shape[0] < 4:
        img = img.permute(1,2,0)

    if not islabel:
        if not save_as_grey:
            img = handleSizes(img, rgb_ok=False)
            # The range is 0 to 255. In order that it's not necessary to normalize, we make it RGB.
            img = np.stack((img, img, img), axis = 2)
        else:
            img = (img - img.min()) / (img.max() - img.min())
            img = 254*img.numpy()
            img = np.concatenate([img]*3, -1)

    path_dir = "/".join(path.split("/")[:-1])
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)

    img = Image.fromarray(img.astype('uint8'))
    img.save(path)

def process_Epoch_Stream(line):
    """
    Process a string with: "epoch: XXX, iters: XXX, XXXXXX"
    :param line: String with epoch and iteration numbers.
    :return: Returns the total iteration number
    """
    line = line.replace(" ", "")
    epoch_section = line.split("epoch:")
    epoch_section = epoch_section[-1]
    epoch_section = epoch_section.split(",", 1)
    epoch = int(epoch_section[0])

    iter_section = epoch_section[-1]
    iter_section = iter_section.split("iters:")
    iter_section = iter_section[-1]
    iter_section = iter_section.split(",")
    iter = int(iter_section[0])

    return (epoch, iter)
def set_deterministic(is_deterministic: bool, random_seed: int = 0) -> None:
    """
    Author: P-Daniel Tudosiu
    Sets the torch and numpy random seeds to random_seed and whether CuDNN's behaviour
    is deterministic or not. If it is deterministic big speed penalties might be
    incurred.
    For more details see https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    Args:
        is_deterministic: If CuDNN behaviour is deterministic or not
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if is_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def SkullStrip(image, label,value = 0.0,  epsilon = 0.65, ):

    '''
    Sets everything that's background according to the label to 0 with a Gaussian-filter applied margin.
    :param image:
    :param label:
    :return:
    '''

    image[label[:, 0:1, ...] > epsilon] = value # We zero label parts of the image
    return image

def adapt_size(img, size):

    '''
    Crops an input image so that its size matches the one specified.
    :param img: Input image. Formats accepted: [X, Y, Z, C], [X, Y, C]
    :param size: List with the size per dimension
    :return: Cropped image (or extended)
    '''

    no_channels = False

    img_ = img.copy()

    if len(img_.shape) > 4:
        TypeError("Only three or four dimensional images supported.")
    elif len(img_.shape) == 2:
        no_channels = True
    if len(size) > 3:

        TypeError("Only two or three-dimensional changes supported.")

    initial_shape = img_.shape

    if no_channels:
        img_ = np.expand_dims(img_, axis=-1)

    on_each_side = []
    crop_mode = []
    for sizeitem in size:
        on_each_side.append([])
        crop_mode.append(False)

    for s_ind, s in enumerate(size):
        rem_s = size[s_ind] - img_.shape[s_ind]
        if rem_s < 0:
            crop_mode[s_ind] = True
            rem_s = img_.shape[s_ind] - size[s_ind]

        if rem_s % 2 == 0:  # Even
            on_each_side[s_ind].append(int(rem_s / 2))
            on_each_side[s_ind].append(int(rem_s / 2))
        else:  # Odd
            on_each_side[s_ind].append(int(np.floor(rem_s / 2)))
            on_each_side[s_ind].append(int(np.floor(rem_s / 2)) + 1)

    if len(size) == 2:
        if crop_mode[0]:
            img_ = img_[on_each_side[0][0]: initial_shape[0] - on_each_side[0][1], :, :]
        else:
                img_ = np.pad(img_, ((on_each_side[0][0], on_each_side[0][1]), (0, 0), (0, 0)))

        if crop_mode[1]:
            img_ = img_[:, on_each_side[1][0]: initial_shape[1] - on_each_side[1][1], :]
        else:
                img_ = np.pad(img_, ((0, 0), (on_each_side[1][0], on_each_side[1][1]), (0, 0)))
    else:
        if crop_mode[0]:
            img_ = img_[on_each_side[0][0]: initial_shape[0] - on_each_side[0][1], :, :, :]
        else:
            img_ = np.pad(img_, ((on_each_side[0][0], on_each_side[0][1]), (0, 0), (0, 0), (0, 0)))

        if crop_mode[1]:
            img_ = img_[:, on_each_side[1][0]: initial_shape[1] - on_each_side[1][1], :, :]
        else:
            img_ = np.pad(img_, ((0, 0), (on_each_side[1][0], on_each_side[1][1]), (0, 0), (0, 0)))
        if crop_mode[2]:
            img_ = img_[:,:, on_each_side[1][0]: initial_shape[1] - on_each_side[1][1], :]
        else:
            img_ = np.pad(img_, ((0, 0), (0, 0), (on_each_side[1][0], on_each_side[1][1]), (0, 0)))


    if no_channels:
        img_ = img_[:, :, 0]  # We remove the expand dims

    return img_
def toRGB(image, was_normalized = False):

    n_channels = image.shape[2]
    if n_channels == 1:
        image = np.concatenate((image, image, image), axis = 2)

    if was_normalized:
        image = (image + 1) / 2.0

    image = 255.0*image
    return image.astype(np.uint8)

def write_plot_metrics(metrics, names, filename):

    '''
    Writes the quality metrics on a TXT file.
    :param metrics: Metric name.
    :param names: Names of the metrics you want on the txt file.
    :param filename: Name of the file (with absolute path) where you want to store it.
    :return:
    '''

    if len(metrics.keys()) != len(names.keys()):
        ValueError("Names must be of equal length than the metrics.")

    header_line = ""
    for n in metrics.keys():
        header_line += names[n] + "\t"

    means_ = {}
    stds_ = {}
    for m, value in metrics.items():
        try:
            means_[m] = np.round(np.mean(value), 3)
            stds_[m] = np.round(np.std(value), 3)
        except:
            means_[m] = "Means"
            stds_[m] = "STDS"

    with open(filename, 'w') as f:
        f.write(header_line + "\n")

        n_items = len(metrics[list(metrics.keys())[0]])
        n_l = ""
        for it in range(n_items):
            for m in metrics.keys():
                if m == "sequence":
                    n_l += metrics[m][it] + '\t'
                else:
                    n_l += str(float(np.round(metrics[m][it], 3))) + '\t'
            n_l += "\n"
            f.write(n_l)
            n_l = ""

        for m in means_.keys():
            n_l += str(means_[m]) + '\t'
        n_l += "\n"
        f.write(n_l)

        for m in stds_.keys():
            n_l += str(stds_[m]) + '\t'
        n_l += "\n"
        f.write(n_l)

    f.close()
def readDatasetsNames(file):

    f = open(file)
    f = f.readlines()
    relevant_line = [l for l in f if "Datasets:" in l]
    if len(relevant_line) > 1:
        Warning("Ambiguous TXT file. More than 1 Dataset line")
    relevant_line = relevant_line[0]
    datasets = relevant_line.split("Datasets:")[-1]
    datasets = datasets.split(",")
    final = []
    for dataset in datasets:
        temp = dataset.strip("\n")
        temp = temp.strip(" ")
        final.append(temp.strip("\n"))

    return final
def codeStats(out_data, savename):
    '''
    Plots a whisker plot of the code, mu and variance of the output of the encoder.
    :param z: Code (1D vector of N elements)
    :param mu: Mean (1D vector of N elements)
    :param logvar: Logvar (1D vector of N elements)
    :param savename: Path and name of the PNG to save.
    :return:
    '''
    fig, ax = plt.subplots()
    ax.set_title('Code plot')
    ax.boxplot(out_data, notch=True)
    plt.xticks([1, 2, 3, 4], ["z", "mu", "std", "noise"])
    plt.savefig(savename)
    plt.close('all')

def MMDScore(input_x, input_y):
    '''
    Maximum-Mean Discrepancy Score (MMD) between two sets of images.
    :param input_x, one of the sets (size BxCxHxW)
    :param input_y, the other of the sets (size BxCxHxW)
    As per MMD in: https://github.com/cyclomon/3dbraingen/blob/master/Test.ipynb
    :return: MMD distance distmean, and number of samples, nimgs
    '''

    distmean = 0.0
    input_x = input_x.view(input_x.size(0), -1)
    input_y = input_y.view(input_y.size(0), -1)
    nimgs = input_x.shape[0]
    xx, yy, zz = torch.mm(input_x,input_x.t()), \
                 torch.mm(input_y,input_y.t()), \
                 torch.mm(input_x,input_y.t())

    beta = (1./(nimgs*nimgs))
    gamma = (2./(nimgs*nimgs))

    distmean = beta * (torch.sum(xx)+torch.sum(yy)) - gamma * torch.sum(zz)

    return distmean,nimgs

def oneHotPVM(parcellation_map):
    '''
    Converts a probabilistic segmentation map into a
    :param parcellation_map:
    :return:
    '''
    out_label = torch.zeros_like(parcellation_map).type(parcellation_map.dtype)
    parcellation_map = torch.argmax(parcellation_map, dim = 1)
    out_label = out_label.scatter_(1, parcellation_map.unsqueeze(1), 1.0)

    return out_label
def PVM2Label(parcellation_map):
    if len(parcellation_map.shape) >= 4:
        return torch.argmax(parcellation_map, dim=1).unsqueeze(1)
    elif len(parcellation_map.shape) == 3:
        return torch.argmax(parcellation_map, dim=0).unsqueeze(0)
    else:
        ValueError("Invalid shape. Tensor must be > 3 dim")
def findLCM(samples, batch_size):
    lcm = samples
    while lcm%batch_size != 0:
        lcm += 1
    return lcm

def findDataset(datasets, name_file):
    for ds in datasets:
        if ds in name_file:
            return ds
    return None
def toNii(image, cut, affine):
    '''
    Reorders the channels of the images as a function of the used cut, and outputs a Nifti image.
    A padding is added up and down the image, to match the original size.
    :param image: torch image
    :param cut: string, a (axial), c (coronal) or s (sagittal)
    :param affine: Nifti Affine
    :return:
    '''

    if cut == 'a':
        image = np.transpose(image, (1, 2, 0))
        zeros = np.zeros((image.shape[0], image.shape[1], 1))
        image = np.concatenate((zeros, image, zeros), axis = 2)
    elif cut == 'c':
        image = image.permute(1, 0, 2)
        zeros = np.zeros((image.shape[0], 1, image.shape[2]))
        image =  np.concatenate((zeros, image, zeros), axis = 1)
    elif cut == 's':
        zeros = np.zeros((1, image.shape[1], image.shape[2]))
        image = np.concatenate((zeros, image, zeros), axis = 0)
    else:
        ValueError("The cut can only be a (axial), c (coronal) or s (sagittal).")

    image = np.asarray(image)
    return nib.Nifti1Image(image, affine = affine)

def plot3DHist(save_name, images_dict, n_bins = 75, colors = ['cyan', 'orange'],
               threshold = 0):
    '''
    Plots a histogram per subplot for each key, value (image) passed in kwargs.
    :param save_name: Path and name of the figure.
    :param n_bins: Number of bins used for each histogram.
    :param images_dict: Dict, key is the title of each image, value is the image.
    :param n_bins: Number of bins for the histogram
    :param colors, tuple containing the color for real and fake images
    :param threshold: float, value given to the background, that is left out the histogram.
    Under threshold, everything is ruled out.
    :return:
    '''

    n_subplots = len(list(images_dict.keys()))
    figure = plt.figure(figsize=(5*n_subplots, 6))
    counter = 1
    for key, im in images_dict.items():
        # We take off 0s because they take up a lot of space.
        gt = im[0].flatten()
        gt = gt[gt>threshold]
        syn = im[1].flatten()
        syn = syn[syn>threshold]
        plt.subplot(1, n_subplots, counter)
        plt.hist(gt.flatten().numpy(), bins = n_bins, alpha = 0.5, color = colors[0])
        plt.hist(syn.flatten().numpy(), bins=n_bins, alpha=0.5, color=colors[1])
        plt.title(key)
        plt.legend(["Real", "Fake"])
        counter += 1

    plt.savefig(save_name)
    plt.close(figure)

def plotCodes(code_dict, sequences, datasets, z_dim, savepath, epoch):

    # Colors
    colors_sub = {'green': ['darkgreen', 'lime', 'olive', 'darkolivegreen'],
                  'blue': ['royalblue', 'dodgerblue', 'navy', 'turquoise'],
                  'pink': ['hotpink', 'magenta', 'mediumvioletred', 'lightpink'],
                  'orange': ['darksalmon', 'coral', 'orange', 'peru']}
    markers_sub = ['^', 'o', '+', 'X']
    colors = ['green', 'blue', 'pink', 'orange']

    fig = plt.figure()
    if z_dim == 3:
        axes = fig.add_subplot(111, projection='3d')

    legend_ = []
    for ind_seq, seq in enumerate(sequences):
        for ind_ds, ds in enumerate(datasets):
            if seq + "-" + ds in code_dict.keys():
                codes_ = np.asarray(torch.stack(code_dict[seq+'-'+ds]))
                if z_dim == 3:
                    axes.scatter(xs=codes_[:,0],
                                 ys=codes_[:,1],
                                 zs=codes_[:,2],
                                 cmap=plt.cm.Spectral,
                                 s = 4,
                                 alpha = 0.75,
                                 color=colors_sub[colors[ind_seq]][ind_ds],
                                 marker = markers_sub[ind_ds]
                                 )
                elif z_dim == 2:
                    plt.scatter(xs=codes_[:,0],
                                 ys=codes_[:,1],
                                 cmap=plt.cm.Spectral,
                                 s = 4,
                                alpha = 0.75,
                                 color=colors_sub[colors[ind_seq]][ind_ds],
                                marker = markers_sub[ind_ds])
                legend_.append(seq + '-' + ds)
    plt.legend(legend_)
    plt.title("Codes in epoch %d" %epoch)
    plt.savefig(savepath)
    plt.close(fig)

def plotTile(img, path, title_, x_ticks, y_ticks):

    f, ax = plt.subplots(1, 1, figsize = (4*len(x_ticks), 4*len(y_ticks)))
    extent_x_max = 0.5*len(x_ticks)
    extent_y_max = 0.5*len(y_ticks)
    ax.imshow(img, extent = [-extent_x_max, extent_x_max, -extent_y_max, extent_y_max],
              cmap = 'gray')
    ax.set_xticks(list(np.linspace(-extent_x_max, extent_x_max, len(x_ticks))))
    ax.set_xticklabels(x_ticks)
    ax.set_yticks(list(np.linspace(-extent_y_max, extent_y_max, len(y_ticks))))
    ax.set_yticklabels(y_ticks)
    plt.title(title_)
    plt.savefig(path)
    plt.close(f)

def saveBatchImages(batch, generated, folder):
    '''
    Saves an the real images, labels and generated images in a folder > annotations / images.
    :param batch: dictionary output by the dataset loader.
    :param generated: BxCxHxW tensor generated by the network
    :param folder: folder where you want to store the images and annotations.
    :return:
    '''

    if not os.path.isdir(os.path.join(folder, 'annotations')):
        os.makedirs(os.path.join(folder, 'annotations'))
    if not os.path.isdir(os.path.join(folder, 'images')):
        os.makedirs(os.path.join(folder, 'images'))

    names_label = [l.split("/")[-1].replace(".npz", "_%d" %(batch['slice_no'][l_ind]))
                  for l_ind, l in enumerate(batch['label_path'])]
    names_img = [l.split("/")[-1].replace(".npz", "_%d" %(batch['slice_no'][l_ind]))
                  for l_ind, l in enumerate(batch['image_path'])]

    for b in range(batch['label'].shape[0]):

        label = batch['label'][b,...].detach().cpu().numpy()
        img_real = batch['image'][b, ...].detach().cpu().numpy()
        img_synt = generated[b, ...].detach().cpu().numpy()
        np.savez_compressed(os.path.join(folder, 'annotations', names_label[b]), img=label)
        np.savez_compressed(os.path.join(folder, 'images', names_img[b]+"_REAL.npz"), img=img_real)
        np.savez_compressed(os.path.join(folder, 'images', names_img[b]+"_FAKE.npz"), img=img_synt)

def plot_D_outputs(outputs_D, save_path, namefig, title, dim_cut = -1):
    '''
    Plots the discriminator(s) output(s).
    :param outputs_D: Dictionary containing keys: real (real images), fake (fake images), pred_real and pred_fake
    (predictions from the discriminator)
    :param save_path: path to save the images
    :param namefig: name of the figure (including extension)
    :param title: title to be given to the figure.
    :return:
    '''

    # We save 1) input image 2) input semantic (as parcellation file) 3) output of all discriminators

    gt = outputs_D['real']
    fake = outputs_D['fake']
    pred_fake = outputs_D['pred_fake']
    pred_real = outputs_D['pred_real']

    n_rows = gt.shape[0]
    n_columns = 2 + len(pred_real)*2
    f = plt.figure(figsize=(5 * n_columns, 3 * n_rows + 4))

    subplot_ctr = 1

    disc_outs = {}
    max_ = -1000
    min_ = 1000

    for d_ind, d in enumerate(pred_real):
        for b in range(n_rows):
            if b not in disc_outs.keys():
                disc_outs[b] = {'fake': [], 'real': []}
            # middle slice
            pred_fake_plot, slice_no = chopSlice(pred_fake[d_ind][b, ...], cut = -1, mode='middle')
            pred_real_plot, _ = chopSlice(d[b, ...], cut = -1, slice_no=slice_no)
            disc_outs[b]['fake'].append(util.tensor2im(pred_fake_plot.squeeze(), is2d=True))
            disc_outs[b]['real'].append(util.tensor2im(pred_real_plot.squeeze(), is2d=True))
            # Gather maximums and minimums
            max_here = max(disc_outs[b]['fake'][-1].max(), disc_outs[b]['real'][-1].max())
            min_here = min(disc_outs[b]['fake'][-1].min(), disc_outs[b]['real'][-1].min())
            if max_here > max_:
                max_  = max_here
            if min_here < min_:
                min_ = min_here

    for b in range(n_rows):

        img_real, slice_no = chopSlice(gt[b, ...], cut = -1,  mode='middle')
        img_real = img_real[0,...]
        img_fake, _ = chopSlice(fake[b, ...], cut = -1, slice_no=slice_no)
        img_fake = img_fake[0,...]
        img_real = util.tensor2im(img_real, is2d=True)
        img_fake = util.tensor2im(img_fake, is2d=True)

        plt.subplot(n_rows, n_columns, subplot_ctr)
        plt.imshow(img_real, cmap='Greys')
        plt.axis('off')
        plt.title("Real")
        subplot_ctr += 1
        for d_ind, d in enumerate(disc_outs[b]['real']):
            plt.subplot(n_rows, n_columns, subplot_ctr)
            plt.imshow(d[..., 1], cmap='jet', vmin=min_, vmax=max_)
            plt.colorbar(fraction=0.035)
            plt.axis('off')
            plt.title("Discriminator %d" % (d_ind))
            subplot_ctr += 1
        plt.subplot(n_rows, n_columns, subplot_ctr)
        plt.imshow(img_fake, cmap='Greys')
        plt.axis('off')
        plt.title("Fake")
        subplot_ctr += 1
        for d_ind, d in enumerate(disc_outs[b]['fake']):
            plt.subplot(n_rows, n_columns, subplot_ctr)
            plt.imshow(d[..., 1], cmap='jet', vmin=min_, vmax=max_)
            plt.colorbar(fraction=0.035)
            plt.axis('off')
            plt.title("Discriminator %d" % (d_ind))
            subplot_ctr += 1

    plt.tight_layout()
    plt.suptitle(title, fontsize=18)
    plt.savefig(os.path.join(save_path, namefig))
    plt.close(f)

def to_one_hot(im, n_labels, lesion_id = None, lesion_values = []):

    '''
    Convert label im into a one_hot encoded version, where there are as many lesion channels as defined in lesion_dict,
    and assigns the lesion channel to the one indexed via lesion_id.
    :param im:
    :param n_labels:
    :param lesion_id: name of the lesion. If no lesion, keep to None.
    :param lesion_values: dict, mapping lesion to its value in the label with one channel. If no lesions, keep to None.
    :return:
    '''

    lesion_dicts = {'wmh': n_labels - 2, 'tumour': n_labels - 1}
    if len(lesion_id) != 0:
        if len(lesion_id) > 1:
            # Multiple lesions
            for l_ind, l in enumerate(lesion_id):
                lesion_id[l_ind] = l.lower()
        else:
            lesion_id = lesion_id[0].lower()
        for key, value in lesion_values.items():
            if key not in lesion_dicts.keys() and key != 'generic':
                ValueError("Lesion type %s has been passed as a channel value, but it's not"
                           "contemplated in this scenario. Please, insert %s in lesion_dicts,"
                           "and make sure that SPADE and its mapping CSV are prepared to take "
                           "that lesion type." %(key, key))


    out = np.zeros(list(im.shape)+[n_labels])
    for l in range(n_labels):
        if l >= n_labels - (len(lesion_dicts.keys())):  # Lesion
            if len(lesion_id) != 0:
                if type(lesion_id) == list:
                    for l_id in lesion_id:
                        out[..., lesion_dicts[l_id]] = im == lesion_values[l_id]
                else:
                    out[..., lesion_dicts[lesion_id]] = im == lesion_values[lesion_id] # The lth value in im corresponds to the lesion, which needs to go in the lesion dict channel.
        else:
            out[..., l] = im == l
    return out


def get_rgb_colours():
    # generated from https://medialab.github.io/iwanthue/
    RGB_tuples2 = np.array([[0,0,0],
                            [57, 65, 195],
                            [200, 180, 100],
                            [0, 166, 47],
                            [225, 79, 207],
                            [151, 217, 64],
                            [255, 255, 0],
                            [255, 102, 0],
                            [255, 51, 0],
                            [255, 204, 102],
                            [69, 105, 242],
                            [169, 213, 58],
                            [255, 170, 198],
                            [1, 221, 115],
                            [164, 40, 176],
                            [64, 197, 69],
                            [190, 204, 21],
                            [1, 82, 210],
                            [161, 195, 9],
                            [108, 78, 213],
                            [119, 209, 63],
                            [176, 78, 213],
                            [74, 178, 35],
                            [211, 117, 255],
                            [40, 154, 0],
                            [253, 123, 255],
                            [1, 103, 229],
                            [88, 165, 0],
                            [231, 122, 255],
                            [0, 220, 137],
                            [167, 0, 149],
                            [146, 217, 95],
                            [128, 39, 160],
                            [173, 212, 79],
                            [174, 124, 255],
                            [127, 159, 0],
                            [133, 128, 255],
                            [220, 199, 51],
                            [0, 136, 253],
                            [242, 147, 0],
                            [2, 104, 213],
                            [184, 162, 0],
                            [159, 131, 255],
                            [59, 130, 0],
                            [244, 76, 196],
                            [0, 151, 59],
                            [252, 58, 167],
                            [0, 125, 37],
                            [228, 0, 135],
                            [40, 223, 182],
                            [243, 22, 123],
                            [0, 179, 124],
                            [228, 0, 120],
                            [0, 139, 65],
                            [210, 0, 128],
                            [123, 218, 143],
                            [163, 0, 122],
                            [153, 215, 118],
                            [85, 65, 169],
                            [199, 205, 73],
                            [65, 72, 168],
                            [255, 172, 53],
                            [23, 78, 170],
                            [247, 189, 68],
                            [45, 156, 255],
                            [224, 96, 2],
                            [1, 160, 234],
                            [221, 69, 27],
                            [22, 217, 246],
                            [224, 17, 61],
                            [0, 191, 160],
                            [250, 34, 112],
                            [0, 152, 106],
                            [254, 46, 102],
                            [1, 205, 208],
                            [239, 56, 61],
                            [49, 203, 255],
                            [185, 46, 0],
                            [72, 184, 255],
                            [255, 139, 46],
                            [2, 94, 175],
                            [214, 141, 0],
                            [154, 144, 255],
                            [137, 150, 0],
                            [212, 134, 255],
                            [94, 132, 0],
                            [255, 117, 223],
                            [0, 98, 21],
                            [253, 151, 255],
                            [57, 101, 0],
                            [255, 101, 183],
                            [102, 121, 0],
                            [108, 58, 156],
                            [217, 199, 83],
                            [126, 50, 137],
                            [143, 136, 0],
                            [122, 163, 255],
                            [183, 140, 0],
                            [108, 177, 255],
                            [255, 90, 64],
                            [1, 177, 187],
                            [187, 0, 36],
                            [110, 217, 189],
                            [186, 0, 114],
                            [156, 212, 155],
                            [193, 0, 87],
                            [0, 119, 81],
                            [255, 86, 146],
                            [33, 94, 44],
                            [255, 120, 204],
                            [57, 92, 23],
                            [255, 162, 251],
                            [155, 128, 0],
                            [189, 167, 255],
                            [190, 116, 0],
                            [1, 139, 202],
                            [255, 119, 67],
                            [17, 83, 150],
                            [255, 184, 95],
                            [110, 65, 123],
                            [201, 203, 113],
                            [145, 38, 110],
                            [0, 115, 92],
                            [255, 89, 126],
                            [25, 94, 61],
                            [255, 106, 79],
                            [158, 190, 255],
                            [158, 36, 2],
                            [94, 138, 187],
                            [177, 96, 0],
                            [240, 175, 255],
                            [84, 86, 0],
                            [255, 158, 215],
                            [166, 112, 0],
                            [82, 101, 151],
                            [155, 58, 0],
                            [188, 170, 224],
                            [146, 77, 0],
                            [224, 184, 238],
                            [128, 89, 0],
                            [160, 12, 92],
                            [149, 167, 113],
                            [173, 0, 66],
                            [85, 97, 46],
                            [255, 125, 173],
                            [97, 81, 30],
                            [247, 178, 216],
                            [155, 39, 28],
                            [176, 167, 112],
                            [164, 16, 54],
                            [232, 192, 130],
                            [133, 54, 99],
                            [255, 183, 116],
                            [137, 99, 144],
                            [255, 161, 106],
                            [154, 34, 74],
                            [239, 188, 137],
                            [143, 50, 67],
                            [255, 173, 160],
                            [118, 72, 25],
                            [227, 157, 184],
                            [128, 65, 37],
                            [255, 162, 173],
                            [181, 137, 97],
                            [255, 122, 140],
                            [170, 104, 130],
                            [255, 142, 107],
                            [195, 123, 133],
                            [255, 157, 135],
                            [174, 106, 101],
                            [255, 138, 160]])
    return RGB_tuples2

def get_colors():
    colors = np.array([[0, 0, 0],
                       [0, 102, 204], # lighter blue
                       [0, 0, 153], # darkish blue
                       [51, 102, 255], # even darkish blue
                       [102, 102, 153],
                       [153, 153, 255],
                       [255, 255, 0],
                       [255, 102, 0],
                       [201, 14, 64],
                       [102, 0, 255]
                       ])
    return colors

def round_and_cap(img, top = 255):
    '''
    Makes sure that the image values stay within value 0 and value top, and are rounded.
    :param img:
    :param top:
    :return:
    '''

    out = img.round()
    out[out<0] = 0.0
    out[out>top] = top
    return out

def translateFromSPADE(label):

    out_label = np.zeros_like(label)
    path_to_csv = "/media/vf19/BigCrumb/Datasets/translator_parc_spadepv.csv"
    mapping_channels = {}
    with open(path_to_csv) as csv_file:
        levels = csv.reader(csv_file, delimiter = ",")
        for row_ind, row in enumerate(levels):
            if row_ind != 0:
                mapping_channels[int(row[1])] = int(row[2])
    for channel_no in range(label.shape[-1]):
        out_label[..., mapping_channels[channel_no]] = label[..., channel_no]

    return out_label

def blurSTD(label, num_threshold = 30, minimal_area = 10, ignore_last = 2):
    '''
    For a one hot label, goes through each of their channels.
    If the number of measurements (clusters) is above threshold, we median filter it.
    :param label: HxWx1xC (C: Channels) numpy array
    :param threshold: threshold above which we perform the blur. Number of clusters
    (output of scipy measurements)
    :param: minimal-area: if the area of the biggest cluster is below this, also blur.
    :param ignore_last: last N channels you ignore: in SPADE, lesions go in the end.
    Lesions can be small and you might not want to filter them because they might wear off.
    This makes sure they are ignored by the function. If there are 3 lesions, ignore_last = 3.
    :return:
    '''

    out_label = np.zeros_like(label)
    for i in range(0, label.shape[-1] - ignore_last):
        lw, num_clusters = scipy.ndimage.measurements.label(label[..., 0, i])
        area = scipy.ndimage.measurements.sum(label[..., 0, i], lw, index=np.arange(lw.max() + 1)).max()
        if num_clusters > num_threshold or area < minimal_area:
            out_label[..., i] = scipy.ndimage.median_filter(label[..., i], size = 3) * label[..., i]
        else:
            out_label[..., i] = label[..., i]
    for i in range(label.shape[-1]-ignore_last, label.shape[-1]):
        out_label[..., i] = label[..., i]

    # We call impute to interpolate holes created by the median filter
    out_label = impute(out_label)

    # Filter to convert to PV map
    out_label = out_label.astype(float)
    # for channel in range(out_label.shape[-1]):
    #     sub_label = scipy.ndimage.gaussian_filter(out_label[...,0,channel], 0.9, 0)
    #     out_label[..., 0, channel] = sub_label

    return out_label


def impute(label_in):
    """ Script to fil the labels pixels that have no channel assign to them.
    Based on:
    https://stackoverflow.com/questions/41550979/fill-holes-with-majority-of-surrounding-values-python
    label: numpy array, HxWx1xC (C: channels)
    """

    # Get pixels that do not have a value in any channel
    mask = np.sum(label_in, axis=-1)
    mask = mask == 0
    # Convert on-hot encoding to single channel image
    label_sc = np.argmax(label_in, axis=-1)

    imputed_label = np.copy(label_sc)
    labels, count = label(mask)
    for idx in range(1, count + 1):
        hole = labels == idx
        surrounding_values = label_sc[binary_dilation(hole) & ~hole]
        most_frequent = Counter(surrounding_values).most_common(1)[0][0]
        imputed_label[hole] = most_frequent

    out_label = (np.arange(label_in.shape[-1]) == imputed_label[..., None]).astype(int)
    return out_label

def arrange_tile(ims, ):
    n_ims = len(ims)
    n_sq = int(np.ceil(np.sqrt(n_ims)))
    tile = []
    for i in range(n_sq):
        subtile = []
        for j in range(n_sq):
            if i * n_sq + j < len(ims):
                subtile.append(ims[i*n_sq+j])
            else:
                subtile.append(np.zeros_like(ims[0]))
        subtile = np.concatenate(subtile, 1)
        tile.append(subtile)
    tile = np.concatenate(tile, 0)
    return tile

def mia_plots(out_ims, top_up_batch = 8, bound_normalization = False):
    '''
    For outputs from Mia Trainer (model_inversion_attack > mia_training_functions),
    plots a batch in a tiled manner.
    :param out_ims: dictionary with:
        - key: real, value: Bx1xHxW tensor
        - key: decoded, value: Bx1xHxW tensor
    :return: 2 rows with the real images on the top and the decoded ones on the bottom.
    '''

    # {'real': data_i['image'].detach().cpu(), 'decoded': x_h.detach().cpu()}

    tiles = []
    for b in range(min(out_ims['real'].shape[0], top_up_batch)):
        sub_tile = []
        sub_tile.append(util.tensor2im(out_ims['real'][b, ...], bound_normalization = bound_normalization))
        sub_tile.append(util.tensor2im(out_ims['decoded'][b, ...], bound_normalization = bound_normalization))
        sub_tile = np.concatenate(sub_tile, 0)
        tiles.append(sub_tile)
    tiles = np.concatenate(tiles, 1)
    return tiles

def retrieve_disease_mask(input_semantics, disease_flags = [6, 7, 8, 9, 10, 11]):

    '''
    For a specific semantic map (BxCxSPATIAL_DIMS) creates a mask for which the disease channels
    are more than 0.5
    :param input_semantics: BxCxSPATIAL_DIMS tensor
    :param disease_flags: indices on the channel dimension (1) that are disease.
    :return: tensor of size Bx1xSPATIAL_DIMS along which disease is 1.0
    '''

    out = input_semantics[:, np.asarray(disease_flags), ...].round()
    out = out.sum(1).unsqueeze(1).clip(0, 1) # Multiply boolean along channel dimension and get back to BxCxHxW...
    return out


def chopSlice(volume, cut, mode = 'middle', slice_no = None):

    cutdic = {'a': -1, 'c': -2, 's': -3, -1: -1, -2: -2, -3: -3}
    if cut not in [-1, -2, -3, 'a', 'c', 's']:
        ValueError("Cut must be a string: a, c or s (axial, coronal or sagittal)"
                   "or a number indicating the dimension of the required cut, "
                   "-1, -2, -.3 Negative indices are required to avoid confusion"
                   "with batches and channels.")
    if mode not in ['middle', 'random']:
        ValueError("Allowed modes are middle or random ONLY.")

    # Slice selection
    if slice_no is None:
        if mode == 'middle':
            slice_no = min(max(volume.shape[cutdic[cut]] // 2, 0), volume.shape[cutdic[cut]] - 1)
        else:
            slice_no = min(max(np.random.randint(volume.shape[cutdic[cut]]), 0), volume.shape[cutdic[cut]] - 1)

    if cutdic[cut] == -1:
        return volume[..., slice_no], slice_no
    elif cutdic[cut] == -2:
        return  volume[..., slice_no, :], slice_no
    elif cutdic[cut] == -3:
        if len(volume.shape) == 3:
            return volume[slice_no, ...], slice_no
        else:
            return volume[..., slice_no, :, :], slice_no


def saveNiiGrid(inputs: torch.Tensor,
                downsample:int = 1,
                grid_shape: list = []):

    """
    Takes a batch of images and puts them in a grid.
    :param inputs: tensor of images of shape B x C x H x W x D
    :param downsample: int, if you want to downsample the grid to avoid extremely large dimensions.
    :param grid_shape: list, if empty, the grid is adjusted depending on the inputs shape.
    Otherwise, it must be a 2 element list (height and width)
    :return:
    """
    if len(grid_shape) != 2:
        grid_shape = [int(np.ceil(np.sqrt(inputs.shape[0])))]*2
    else:
        grid_shape = grid_shape

    counter = 0
    grid_cols = []
    for c_col in range(grid_shape[0]):
        grid_row = []
        for c_row in range(grid_shape[1]):
            if counter < inputs.shape[0]:
                img_ = inputs[counter, ...]
                if img_.shape[0] > 1:
                    img_ = torch.argmax(img_, 0)
                else:
                    img_ = img_.squeeze(0)
                grid_row.append(img_.type(torch.float))
            else:
                grid_row.append(torch.zeros_like(img_))
            counter +=1
        grid_row = torch.cat(grid_row, 1)
        grid_cols.append(grid_row)
    grid = torch.cat(grid_cols, 0)

    if downsample > 1:
        new_shape = [i//downsample for i in grid.shape]
        grid = grid.unsqueeze(0).unsqueeze(0)
        grid = torch.nn.functional.interpolate(grid, new_shape)
        grid = grid.squeeze(0).squeeze(0)

    return grid


def getRGBGrid(inputs: torch.Tensor,
                n_slices:int = 10,
                label_index: int = None,
                ):

    '''
    Creates a grid of axial slices taken from different tensors.
    :param inputs: Images, as a tensor of BxCxHxWxD.
    :param n_slices:
    :param label_index
    :return:
    '''

    all_columns = []
    all_slices = list(range(inputs.shape[-1]))[::n_slices]
    if label_index is not None:
        label_column = []
    for sl in all_slices:
        column = []
        for b in range(inputs.shape[0]):
            img_slice = inputs[b, 0, ..., sl].numpy()
            if b == label_index:
                img_slice = get_colors()[img_slice.astype(int)] # Ends up being a H x W x 3
            if label_index is not None and b == label_index:
                label_column.append(img_slice)
            else:
                column.append(img_slice)
        all_columns.append(np.concatenate(column, 1))

    label_column = np.concatenate(label_column, 0)
    all_columns = np.concatenate(all_columns, 0)
    if label_index is not None:
        return label_column, all_columns
    else:
        return all_columns

def reconvert_styles(styles, cut, crop_shape, careless_last = False):
    if careless_last:
        return torch.stack([styles]*crop_shape[-1], -1)
    else:
        if cut == 'a':
            n_repetitions = crop_shape[-1]
            return torch.stack([styles]*crop_shape[-1], -1)
        elif cut == 'c':
            n_repetitions = crop_shape[1]
            return torch.stack([styles] * n_repetitions, -2)
        elif cut == 's':
            n_repetitions = crop_shape[0]
            return torch.stack([styles] * n_repetitions, -3)