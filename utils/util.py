"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import importlib
import torch
from argparse import Namespace
import numpy as np
from PIL import Image
import os
import argparse
import dill as pickle
import utils.coco


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# returns a configuration for creating a generator
# |default_opt| should be the opt of the current experiment
# |**kwargs|: if any configuration should be overriden, it can be specified here


def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, is2d = False,
              no_channel = False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype))
        return image_numpy
    if is2d:
        b_dim = 4
    else:
        b_dim = 5
    if image_tensor.dim() == b_dim:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image, is2d=is2d)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)

        return images_np

    if image_tensor.dim() < 3: # 3D or 2D
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()

    if b_dim == 5:
        transpose_axes = (1,2,3,0)
    else:
        transpose_axes = (1,2,0)

    image_numpy = 255 * (np.transpose(image_numpy, transpose_axes) - image_numpy.min()) / (
            image_numpy.max() - image_numpy.min())
    image_numpy = np.clip(image_numpy, 0, 255)

    if image_numpy.shape[-1] == 1:
        if no_channel:
            image_numpy = np.squeeze(image_numpy, -1)
        else:
            image_numpy = np.concatenate((image_numpy, image_numpy, image_numpy), axis = -1)

    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, is2d = False):

    if is2d:
        b_dim = 4
    else:
        b_dim = 5
    if label_tensor.dim() == b_dim:
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype, is2d=is2d)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        return images_np

    label_tensor = label_tensor.cpu().numpy()
    if label_tensor.shape[0] > 1: # Channels
        label_tensor = np.expand_dims(np.argmax(label_tensor, 0), 0)

    if is2d:
        transpose_axes = (1,2,0)
    else:
        transpose_axes = (1,2,3,0)

    label_numpy = np.transpose(label_tensor, transpose_axes)
    result = label_numpy.astype(imtype)
    return result

def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt, device):

    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available() or opt.use_ddp:
        net.to(device)

def load_network(net, label, epoch, opt, strict = True):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights, strict = strict)
    del weights
    return net

def save_optimizer(optimizer, label, epoch, opt):

    save_filename = '%s_optimizer_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(optimizer.state_dict(), save_path)

def load_optimizer(optimizer, label, epoch, opt, strict = True):
    save_filename = '%s_optimizer_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    if os.path.exists(save_path):
        weights = torch.load(save_path)
        optimizer.load_state_dict(weights)
    del weights
    return optimizer

def load_network_from_file(net, file, strict = True):

    weights = torch.load(file)
    net.load_state_dict(weights, strict = strict)
    return net

def preprocess_input(data, name_mod2num, device, label_nc, contain_dontcare_label = False):
    # create one-hot label map
    label_map = data['label'].as_tensor().to(device)
    image = data['image'].as_tensor().to(device)
    style = data['style_image'].as_tensor().to(device)
    bs, _, h, w, d = label_map.size()
    nc = label_nc + 1 if contain_dontcare_label \
        else label_nc # Adds 1 if there is a "dontcare" label
    input_semantics = label_map
    modalities = torch.tensor([name_mod2num[i] for i in data['this_seq']]).type(torch.float).to(device)

    return input_semantics, image, style, modalities

def colors(n_col = 11):
    colors= np.array([[0, 0, 0],
                     [0, 102, 204],
                     [0, 0, 153],
                     [51, 102, 255],
                     [102, 102, 153],
                     [153, 153, 255],
                     [255, 255, 0],
                     [255, 14, 64],
                     [255, 102, 0],
                     [201, 14, 64],
                     [102, 0, 255],
                     ]
                     )

    if n_col > 11:
        while len(colors < n_col):
            new_color = [np.random.randint(255),
                         np.random.randint(255),
                         np.random.randing(255)]
            if new_color not in colors:
                colors.append(new_color)
    return colors

