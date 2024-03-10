from __future__ import annotations
import numpy as np
import os
import nibabel as nib
import monai
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from data.dataset import Spade3DSet
from moreutils import getRGBGrid, reconvert_styles, structural_Similarity
import torch
from tqdm import tqdm
from models_spade.pix2pix_model import Pix2PixModel
import json
from utils.util import preprocess_input
from skimage.metrics import structural_similarity

def cross_styles(image_generator, opt):

    # Modify the options regarding datasets
    opt.non_corresponding_dirs = True
    opt.non_corresponding_ims = False

    for ind_style, style in enumerate(opt.style_names):
        opt.style_dict = opt.style_dicts_cross_styles[ind_style]
        # Create datasets
        spade_dataset = Spade3DSet(opt)
        # Create directories
        results_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.root_results_dir, style)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        if not os.path.isdir(os.path.join(results_dir, 'cross_styles')):
            os.makedirs(os.path.join(results_dir, 'cross_styles'))

        # We generated images
        loader = monai.data.dataloader.DataLoader(spade_dataset.test_set,
                                                  batch_size=6, num_workers=2,
                                                  shuffle=False, drop_last=False)
        test_image_index = 0
        for data_i in tqdm(loader, desc = "Cross-styles test..."):
            input_label, input_image, input_style, mods = image_generator.preprocess_input(data_i)
            generated = image_generator(input_label, input_image, input_style, mods,
                                        mode = 'inference')
            reconverted_style = reconvert_styles(input_style,
                                                 options.cut,
                                                 options.crop_size,
                                                 careless_last=True)
            for b in range(generated.shape[0]):
                label, img_grid = getRGBGrid(torch.stack([torch.argmax(input_label[b,...], 0).unsqueeze(0).detach().cpu(),
                                               input_image[b, ...].detach().cpu(),
                                               reconverted_style[b, ...].detach().cpu(),
                                               generated[b, ...].detach().cpu()],
                                              0),
                                  n_slices=10,
                                  label_index=0)

                f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(4, 6))
                ax1.imshow(label); ax1.axis('off');
                ax2.imshow(img_grid, cmap="gist_gray"); plt.axis('off');
                plt.subplots_adjust(hspace=0.0, wspace=0.0)
                plt.tight_layout();
                plt.suptitle("Sample %d (%s)" %(test_image_index, style))
                plt.savefig(os.path.join(results_dir, 'cross_styles', "cross_styles_%s_%d.png" %(style,
                                                                                                 test_image_index)))
                plt.close(f)
                test_image_index +=1

def test_set_and_ssim_per_tissue(image_generator, opt):
    channel_names = ['csf', 'gm', 'wm', 'dgm', 'brainstem', 'wmh', 'tumour', 'edema', 'gdtumour']
    output_dict = {}
    # Modify the options regarding datasets
    opt.style_dict = None
    opt.non_corresponding_dirs = False
    opt.non_corresponding_ims = False
    opt.max_dataset_size = opt.max_images
    # Create datasets
    spade_dataset = Spade3DSet(opt)

    # Create directories
    results_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.root_results_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if not os.path.isdir(os.path.join(results_dir, 'test_set_ssim')):
        os.makedirs(os.path.join(results_dir, 'test_set_ssim'))

    # We generated images
    for mod in opt.sequences:
        dataset = spade_dataset.resetSeqs(mod)
        loader = monai.data.dataloader.DataLoader(dataset,
                                                  batch_size=6, num_workers=2,
                                                  shuffle=False, drop_last=False)

        for data_i in tqdm(loader, desc="Test-set SSIM test..."):
            input_label, input_image, input_style, _, mods = preprocess_input(data_i,
                                                                           device="cuda" if torch.cuda.is_available() else "cpu",
                                                                           name_mod2num=image_generator.name_mod2num,
                                                                           label_nc=opt.label_nc,
                                                                           contain_dontcare_label=opt.contain_dontcare_label)
            generated = image_generator(input_label, input_image, input_style, mods,
                                        mode='inference')
            # Main metrics on just one
            for b in range(generated.shape[0]):
                input_image_path = data_i['image_meta_dict']['filename_or_obj'][b]
                input_style_path = data_i['style_image_meta_dict']['filename_or_obj'][b]
                gt_img = input_image[b, ...].detach().cpu().squeeze()
                gen_img = generated[b, ...].detach().cpu().squeeze()
                lab = input_label[b, ...].detach().cpu()
                ssim_values = {'global': structural_similarity(gt_img.numpy(), gen_img.numpy())}

                # We need to scale
                ssim_values = {'global': float(structural_similarity(gt_img.numpy(), gen_img.numpy()))}
                mses_values = {'global': float(np.sqrt((gt_img.numpy() - gen_img.numpy()) ** 2).mean())}
                for tissue in range(1, lab.shape[0]):
                    # We mask the images
                    gt_masked = gt_img * (lab[tissue, ...] > 0.5)
                    gen_masked = gen_img * (lab[tissue, ...] > 0.5)
                    ssim_values[channel_names[tissue-1]] = structural_similarity(gt_masked.numpy(),
                                                                               gen_masked.numpy())
                    mses_values[channel_names[tissue-1]] =  float(np.sqrt((gt_masked.numpy() -
                                                                           gen_masked.numpy()) ** 2).mean())
                output_dict[input_image_path.split("/")[-1]] = {
                    'input_image': input_image_path,
                    'input_style': input_style_path,
                    'ssim_values': ssim_values,
                    'mses_values': mses_values
                }

        with open(os.path.join(results_dir, 'test_set_ssim', 'ssims_per_tissue_%s.json' %mod), 'w') as f:
            json.dump(output_dict, f, indent=4)
            f.close()

def test_set_and_ssim(image_generator, opt,):

    # Modify the options regarding datasets
    opt.style_dict = None
    opt.non_corresponding_dirs = False
    opt.non_corresponding_ims = False
    opt.max_dataset_size = opt.max_images

    # Create datasets
    spade_dataset = Spade3DSet(opt)

    # Create directories
    results_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.root_results_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if not os.path.isdir(os.path.join(results_dir, 'test_set_ssim')):
        os.makedirs(os.path.join(results_dir, 'test_set_ssim'))

    # We generated images
    dataset = spade_dataset.getDatasets()
    loader = monai.data.dataloader.DataLoader(dataset,
                                              batch_size=6, num_workers=2,
                                              shuffle=False, drop_last=False)
    test_image_index = 0
    quality_metrics = {'ssim': [], 'mse': [], 'sd': []}
    for mod in options.sequences:
        dataset = spade_dataset.resetSeqs(mod)
        loader = monai.data.dataloader.DataLoader(dataset,
                                                  batch_size=6, num_workers=2,
                                                  shuffle=False, drop_last=False)
        for data_i in tqdm(loader, desc="Test-set SSIM test..."):
            input_label, input_image, input_style, _, mods = preprocess_input(data_i,
                                                                           device="cuda" if torch.cuda.is_available() else "cpu",
                                                                           name_mod2num=image_generator.name_mod2num,
                                                                           label_nc=opt.label_nc,
                                                                           contain_dontcare_label=opt.contain_dontcare_label)
            all_generated = []
            for regeneration_turn in range(options.regeneration_index):
                generated = image_generator(input_label, input_image, input_style, mods,
                                            mode='inference')
                for b in range(generated.shape[0]):
                    quality_metrics['ssim'].append(structural_similarity(
                        input_image[b, 0, ...].detach().cpu().numpy(),
                        generated[b,0, ...].detach().cpu().numpy()))
                    quality_metrics['mse'].append(np.sqrt((input_image[b, ...].detach().cpu().numpy() - \
                                                           generated[b, ...].detach().cpu().numpy()) ** 2).mean())

                # Multiple instances
                if len(all_generated) == 0:
                    if options.chunk_size is None:
                        crop_size_ = options.crop_size
                    else:
                        crop_size_ = options.crop_size[:-1]+[options.chunk_size]
                    reconverted_style = reconvert_styles(input_style,
                                                         options.cut,
                                                         crop_size_,
                                                         careless_last=True)
                    for b in range(generated.shape[0]):
                        label, img_grid = getRGBGrid(torch.stack([
                            torch.argmax(input_label[b, ...], 0).unsqueeze(0).detach().cpu(),
                            input_image[b, ...].detach().cpu(),
                            reconverted_style[b, ...].detach().cpu(),
                            generated[b, ...].detach().cpu()],
                            0),
                            n_slices=10,
                            label_index=0)
                        f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]}, figsize=(8,12))
                        ax1.imshow(label); ax1.axis('off'); plt.axis('off');
                        ax2.imshow(img_grid, cmap="gist_gray"); plt.axis('off');
                        plt.subplots_adjust(hspace=0.0, wspace=0.0)
                        plt.tight_layout();
                        plt.suptitle("Label, input image, style, generated")
                        plt.savefig(os.path.join(results_dir, 'test_set_ssim', "test_set_ssim_%s_%d.png" %(mod, test_image_index)))
                        plt.close(f)
                        test_image_index += 1

                        quality_metrics['ssim'].append(structural_Similarity(input_image[b, ...].detach().cpu(),
                                                     generated[b, ...].detach().cpu(), mean=True))
                        quality_metrics['mse'].append(np.sqrt((input_image[b, ...].detach().cpu().numpy() - \
                                                                generated[b, ...].detach().cpu().numpy())**2).mean())

                all_generated.append(generated)

            all_generated = torch.stack(all_generated, 0)
            for b in range(all_generated.shape[1]):
                quality_metrics['sd'].append(torch.std(all_generated[:, b, ...]).mean().item())

        # Now that we have the metrics, we plot:
        boxprops = dict(color='#0066cc')
        flierprops = dict(marker='.', markersize=4, markeredgecolor='#004d99')
        medianprops = dict(linestyle='-.', linewidth=2, color='#001a33')
        f = plt.figure()
        data = [quality_metrics['ssim'], quality_metrics['mse'], quality_metrics['sd']]
        plt.boxplot(data, boxprops=boxprops, flierprops=flierprops,medianprops=medianprops)  # Or you can use the boxplot from Pandas
        plt.xticks([1,2,3], list(quality_metrics.keys()))
        for i in [1, 2, 3]:
            y = data[i - 1]
            x = np.random.normal(i, 0.02, len(y))
            plt.plot(x, y, 'c.', alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'test_set_ssim', 'ssim_test_results_%s.png' %mod))
        plt.close(f)

        print("%s --- SSIM: %.3f +- %.3f, MSE: %.3f +- %.3f" %(mod, np.mean(quality_metrics['ssim']),
                                                        np.std(quality_metrics['ssim']),
                                                        np.mean(quality_metrics['mse']),
                                                        np.std(quality_metrics['mse'])))

options = TestOptions().parse()
device = "cuda" if torch.cuda.is_available() else "cpu"
image_generator = Pix2PixModel(options, device)

if options.test_set_and_ssim:
    test_set_and_ssim(image_generator, options)
if options.cross_styles:
    cross_styles(image_generator, options)

