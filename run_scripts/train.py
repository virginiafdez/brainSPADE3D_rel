import os
import moreutils as uvir
from options.train_options import TrainOptions
import data
import sys
import nibabel as nib
from utils.iter_counter import IterationCounter
from utils.visualizer import Visualizer
from data.dataset_utils import clear_data
import numpy as np
import torch
from trainers.pix2pix_trainer import Pix2PixTrainer
from copy import deepcopy
import gc
from data.dataset import Spade3DSet
import shutil
from monai.utils import set_determinism
from monai.data.dataloader import DataLoader
from utils.tensorboard_writer import BrainspadeBoard
import monai
from tqdm import tqdm
from utils.util import tensor2im, tensor2label
import torch.distributed as dist
from data.dataset_utils import ddpDataDicts
from  monai.data import ThreadDataLoader
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
##
from pynvml.smi import nvidia_smi

from pynvml.smi import nvidia_smi

def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")

# Parse options
opt = TrainOptions().parse()

# Set seed
set_determinism(seed=0)

# Use DDP
# DDP set-up
if opt.use_ddp:
    if "LOCAL_RANK" in os.environ:
        print("Setting up DDP.")
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank != 0:
            f = open(os.devnull, "w")
            sys.stdout = sys.stderr = f
        # initialize the distributed training process, every GPU runs in a process
        dist.init_process_group(backend="nccl", init_method="env://")
        print(dist.get_rank())
        print(dist.get_world_size())
        device = torch.device(f"cuda:{local_rank}")

    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)

# Dataset
dataset_container = Spade3DSet(opt)
train_set, val_set = dataset_container.getDatasets()
if opt.use_ddp:
    #ThreadDataLoader
    loader = monai.data.DataLoader(train_set, shuffle = False, batch_size=opt.batchSize, num_workers=opt.nThreads, )
    loader_val = monai.data.DataLoader(val_set, shuffle=False, batch_size=opt.batchSize, num_workers=opt.nThreads, )
else:
    loader = monai.data.DataLoader(train_set, shuffle=False, batch_size=opt.batchSize, num_workers=opt.nThreads)
    loader_val = monai.data.DataLoader(val_set, shuffle=False, batch_size=opt.batchSize, num_workers=opt.nThreads)

# Initialisation network
trainer = Pix2PixTrainer(opt, device)

# Iterations counter
iter_counter = IterationCounter(opt, len(dataset_container))

# Visualization tool
if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
    # Ony initialised if rank = 0
    visualizer = Visualizer(opt)
    visualizer.initialize_Validation(opt.continue_train)
    # Tensorboard
    if opt.use_tboard:
        tboard = BrainspadeBoard(opt)
    # Validation save ID
    save_im_id = None

# Gradients saved
gradients = {}
activations = {}

# Training Loop
for epoch in iter_counter.training_epochs():
    trainer.record_epoch(epoch)
    iter_counter.record_epoch_start(epoch)
    train_gen_count = 0
    train_dis_count = 0
    train_total = 0
    for dind, data_i in enumerate(tqdm(loader)):
        print_gpu_memory_report()
        train_total += 1
        # Phase 1 Generator training.
        # If accuracy is none, we train.
        # If accuracy is < 65%, we train discriminator only.
        # If accuracy is > 85%, we train generator only.
        # If accuracy is between both, we train both.

        # Ammend dataset container to register stored slices
        d_acc = trainer.get_disc_accuracy()
        if d_acc is None:
            d_acc = iter_counter.add_assess_accuracy(None)
        else:
            d_acc = iter_counter.add_assess_accuracy(d_acc['D_acc_total'].mean().item())
        if d_acc is None:
            train_disc = True
            train_gen = True
        elif trainer.disc_threshold['low'] <= d_acc <= trainer.disc_threshold['up']:
            train_disc = True
            train_gen = True
        elif d_acc < trainer.disc_threshold['low']:
            train_disc = True
            train_gen = False
        elif d_acc > trainer.disc_threshold['up']:
            train_disc = False
            train_gen = True
        else:
            Warning("Non numeric accuracy.")

        iter_counter.record_one_iteration()

        # Phase 1 Generator training.
        if train_gen:
            print("Training the generator on... %s" %device)
            # If train_enc_only < epochs it means that from this epoch onward we only
            # train the encoder. Otherwise, both.
            iter_counter.record_one_gradient_iteration_gen()
            if opt.train_enc_only is not None:
                if epoch >= opt.train_enc_only:
                    trainer.run_encoder_one_step(data_i)
                else:
                    if iter_counter.needs_gradient_calc(for_disc=False) and opt.tboard_gradients:
                        gradients_ = trainer.run_generator_one_step(data_i, with_gradients=True)
                        gradients.update(gradients_)
                    else:
                        trainer.run_generator_one_step(data_i)
            else:
                if iter_counter.needs_gradient_calc(for_disc=False) and opt.tboard_gradients:
                    gradients_ = trainer.run_generator_one_step(data_i, with_gradients=True)
                    gradients.update(gradients_)
                else:
                    trainer.run_generator_one_step(data_i)
            if iter_counter.needs_activations(for_disc=False) and opt.tboard_activations:
                activations.update({'enc_mu': trainer.last_hooks['enc_mu'],
                                    'enc_sigma': trainer.last_hooks['enc_sigma'],
                                    'deocder': trainer.last_hooks['decoder']})

            generated = trainer.get_latest_generated()

            # If display is needed, we save the relevant data.
            data_copy = deepcopy(data_i)

            # Store generator losses
            iter_counter.store_losses(trainer.g_losses, None)

            train_gen_count += 1

        # Part 2. Train the discriminator.
        if train_disc:

            iter_counter.record_one_gradient_iteration_dis()
            if iter_counter.needs_gradient_calc(for_disc=True) and opt.tboard_gradients:
                if opt.topK_discrim:
                    if iter_counter.needs_D_display():
                        gradients_, outputs_D = trainer.run_discriminator_one_step(data_i,
                                                                                   with_gradients=True,
                                                                                   return_predictions=True)
                    else:
                        gradients_ = trainer.run_discriminator_one_step(data_i, with_gradients=True)
                else:
                    if iter_counter.needs_D_display():
                        gradients_, outputs_D = trainer.run_discriminator_one_step(data_i, with_gradients=True,
                                                                                   return_predictions=True)
                    else:
                        gradients_ = trainer.run_discriminator_one_step(data_i, with_gradients=True)
                gradients.update(gradients_)
            else:
                if opt.topK_discrim:
                    if iter_counter.needs_D_display():
                        outputs_D = trainer.run_discriminator_one_step(data_i, return_predictions=True)
                    else:
                        trainer.run_discriminator_one_step(data_i)
                else:
                    if iter_counter.needs_D_display():
                        outputs_D = trainer.run_discriminator_one_step(data_i, return_predictions=True)
                    else:
                        trainer.run_discriminator_one_step(data_i)
            iter_counter.store_losses(None, trainer.d_losses, trainer.d_accuracy)
            train_dis_count += 1

            if iter_counter.needs_activations(for_disc=True) and opt.tboard_activations:
                for key_hook, val_hook in trainer.last_hooks.items():
                    if 'disc' in key_hook:
                        activations.update({key_hook:val_hook})
        # Part 3. Tests and display
        # Part 3-1. Code distribution boxplots are saved in web/code_plots
        if iter_counter.needs_enc_display():
            if opt.type_prior == 'N':
                gen_z, gen_mu, gen_logvar, gen_noise = trainer.run_encoder_tester(data_i)
                visualizer.save_codes(gen_z, gen_mu, gen_logvar, data_i['this_seq'], gen_noise, iter_counter.current_epoch,
                                 iter_counter.epoch_iter)
            elif opt.type_prior == 'S':
                gen_z, gen_mu, gen_logvar = trainer.run_encoder_tester(data_i)
                visualizer.save_codes(gen_z, gen_mu, gen_logvar, sequence = data_i['this_seq'],
                                      epoch = iter_counter.current_epoch,
                                      iter = iter_counter.epoch_iter)
        if iter_counter.needs_D_display():
            if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
                visualizer.plot_D_results(outputs_D, epoch, iter_counter.epoch_iter) # Plot discriminator outputs

        # Clear.
        clear_data(trainer, data_i)

        # Part 3-2. Printing of losses IF:
        # We are every N_print iterations
        # We have latest losses.
        if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
            if iter_counter.needs_printing() and iter_counter.epoch_iter>1:
                if trainer.g_losses is not None:
                    losses = trainer.get_latest_losses_nw()
                    accuracies = trainer.get_disc_accuracy()
                    losses.update(accuracies)
                    visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                         losses, iter_counter.time_per_iter)
                else:
                    print("No losses available at printing time for epoch %d and iteration %d" %(epoch,
                                                                                                 iter_counter.epoch_iter))

        # Part 3-3. We save the last generated set of mages in web/images
        if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
            if iter_counter.needs_displaying() and iter_counter.epoch_iter>1:
                if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
                    if trainer.get_latest_generated() is not None:
                        generated = trainer.get_latest_generated()
                        img_dir = visualizer.img_dir
                        fig_name = os.path.join(img_dir,
                                                "epoch_%s_iter_%s.png" % (epoch, iter_counter.total_steps_so_far))
                        # We append all we want to save in a list
                        if opt.skullstrip:
                            data_copy['style_image'] = uvir.SkullStrip(data_copy['style_image'], data_copy['style_mask'],
                                                                  data_copy['style_image'].min())
                            data_copy['image'] = uvir.SkullStrip(data_copy['image'], data_copy['label'],
                                                            data_copy['image'].min())
                        val_imgs = {
                            'input label': data_copy['label'],
                            'input style': data_copy['style_image'],
                            'input image': data_copy['image'],
                            'generated': generated
                        }

                        uvir.saveFigs(val_imgs, fig_name, create_dir=True, nlabels=opt.label_nc,
                                      sequences=data_copy['this_seq'],
                                      suptitle="Training epoch %d; iter %d" %(epoch, iter_counter.epoch_iter))

        # Part 3-4. We save the iteration stage of the network.
        if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
            if iter_counter.needs_saving():
                if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
                    # Save data
                    print('Saving the latest model (epoch %d, total steps %d)' % (epoch, iter_counter.total_steps_so_far))
                    trainer.save('latest', device = device)
                    iter_counter.record_current_iter()


        # Part 3-5. If gradients, then plot summary them
        if len(gradients) != 0:
            if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
                try:
                    if opt.use_tboard and opt.tboard_gradients:
                        tboard.log_grad_histograms(gradients, epoch, iter_counter.epoch_iter,
                                                   len(dataset_container)*1)
                except:
                    gradients = {}

        # Part 3.6. If activations, then plot summary them
        if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
            if len(activations) != 0:
                if opt.use_tboard and opt.tboard_activations:
                    tboard.log_act_histograms(activations, epoch, iter_counter.epoch_iter,
                                               len(dataset_container)*1)
                activations = {}

    torch.cuda.empty_cache()  # Epoch end. Empty cache

    # Part 4. Validation.
    if iter_counter.needs_testing():
        if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
            print("Validation results:\n")
            if opt.z_dim in [2,3]:
                do_code = True # Plot codes of each validation image in 2D or 3D plot.
            else:
                do_code = False # If the latent dimension is > 3, we don't plot (dimensionality reduction would be required)
            with torch.no_grad():
                # We select only one image from the validation set
                if loader_val.__len__() > 1:
                    if save_im_id == None:
                        save_im_id = np.random.randint(0, loader_val.__len__() - 1)
                else:
                    save_im_id = 0

                # Epoch-wise metrics to-validation
                results_values = []  # Values for the test
                accuracy_mod = []
                accuracy_dat = []
                losses_nw = {}
                losses = {}
                if do_code:
                    codes = {}

                for t, data_t in enumerate(loader_val):
                    if do_code:
                        gen_val, g_losses, g_losses_nw, code_val = trainer.run_tester_one_step(data_t, get_code=True)
                        # Store codes
                        for b in range(code_val.shape[0]):
                            if data_t['this_seq'][b]+"-"+data_t['st_dataset'][b] in codes.keys():
                                codes[data_t['this_seq'][b]+"-"+data_t['st_dataset'][b]].append(code_val[b,...].detach().cpu())
                            else:
                                codes[data_t['this_seq'][b]+"-"+data_t['st_dataset'][b]] = [code_val[b,...].detach().cpu()]
                        # Store accuracies
                        if 'acc_mod' in g_losses_nw.keys() and 'acc_dat' in g_losses_nw.keys():
                            accuracy_mod.append(g_losses_nw['acc_mod'])
                            accuracy_dat.append(g_losses_nw['acc_dat'])
                    else:
                        gen_val, g_losses, g_losses_nw = trainer.run_tester_one_step(data_t)

                    clear_data(None, data_t)

                    # Part 4-1 Save images in web/validation
                    if opt.skullstrip:
                        gt_image = uvir.SkullStrip(data_t['image'].detach().cpu(),
                                                   data_t['label'].detach().cpu(),
                                                   data_t['image'].min())
                    if t == save_im_id:  # We only save one of the instances per epoch
                        fig_name = os.path.join(visualizer.val_dir,
                                                "validation_epoch_%s_%s.png" % ('inference', epoch))
                        if opt.skullstrip:
                            data_t['style_image'] = uvir.SkullStrip(data_t['style_image'], data_t['style_mask'],
                                                               data_t['style_image'].min())
                            data_t['image'] = uvir.SkullStrip(data_t['image'], data_t['label'],
                                                         data_t['image'].min())
                        val_imgs = {
                            'input label': data_t['label'],
                            'input style': data_t['style_image'],
                            'input image': data_t['image'],
                            'generated': gen_val
                        }

                        uvir.saveFigs(val_imgs, fig_name, create_dir=True, nlabels = opt.label_nc,
                                      sequences = data_t['this_seq'],
                                      suptitle="Validation epoch %d" %epoch)

                        # Save full nifti
                        to_save_grid = []
                        for b in range(gen_val.shape[0]):
                            to_save_grid.append(gen_val[b,...].detach().cpu())
                            to_save_grid.append(gt_image[b,...])
                        to_save_grid = torch.stack(to_save_grid, 0)
                        grid_tensor = uvir.saveNiiGrid(to_save_grid, grid_shape=[gen_val.shape[0], 2])
                        nif_grid = nib.Nifti1Image(grid_tensor.numpy(), affine = np.eye(4))
                        nib.save(nif_grid, os.path.join(visualizer.val_dir,"validation_epoch_%s_%s.nii.gz" % ('inference', epoch) ))

                    # Image quality metric
                    ssim_item = 0
                    ssim_item += uvir.structural_Similarity(gt_image, gen_val, mean=True)
                    results_values.append(ssim_item)

                    # Average losses
                    for loss_item, loss_value in g_losses_nw.items():
                        if 'acc_mod' in loss_item or 'acc_dat' in loss_item:
                            # These are treated separately!
                            continue
                        # Unweighted losses
                        if loss_item not in losses_nw.keys():
                            losses_nw[loss_item] = [loss_value]
                        else:
                            losses_nw[loss_item].append(loss_value)
                        # Weighted loss
                        if loss_item not in losses.keys():
                            losses[loss_item] = [g_losses[loss_item]]
                        else:
                            losses[loss_item].append(g_losses_nw[loss_item])

                # Plot codes if requested
                if do_code:
                    uvir.plotCodes(codes, opt.sequences, opt.datasets, opt.z_dim,
                                   os.path.join(visualizer.val_dir, 'code_plots_%d.png' %epoch),
                                   epoch)

                # Process losses and register them
                final_losses = {}
                final_losses_nw = {}
                for loss_item, loss_value_list in losses.items():
                    final_losses[loss_item] = np.mean(loss_value_list)
                    final_losses_nw[loss_item] = np.mean(losses_nw[loss_item])

                visualizer.register_Val_Losses(epoch, errors_nw=final_losses_nw, errors_w=final_losses, print_it=True)

                # Part 4-2 Save structural similarity txt in web/validation
                visualizer.register_Test_Results(epoch, {'SSIM': np.mean(results_values)})
                print("SSIM %.3f\tAcc_mod\n" %(100*np.mean(results_values)))

    # Part 5. We update LR
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    # Part 6. We save the network.
    if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
        if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest', device = device)
        if epoch % opt.save_epoch_copy == 0:
            trainer.save(epoch, device = device)

    if opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp:
        print("Trained generator %d/%d" %(train_gen_count, train_total))
        print("Trained discriminator %d/%d" %(train_dis_count, train_total))

    # Part 7. Summary writer
    if opt.use_tboard and (opt.use_ddp and dist.get_rank() == 0 or not opt.use_ddp):
        tboard.log_results(iter_counter.getStoredLosses(), epoch, is_val=False)
        tboard.log_results(final_losses_nw, epoch, is_val=True)

    # Part 8. Cleaning.
    gc.collect()
    torch.cuda.empty_cache()

print("Removing cache directory...")
shutil.rmtree(dataset_container.cache_dir)
