# Before importing...
import sys
from label_ldm.label_generator.labelgen_utils import checkResolution, log_3d_img, get_lr, log_mlflow,\
    AdversarialReferee
import os
import torch
from monai.losses import FocalLoss
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
import argparse
import numpy as np
import shutil
from pathlib import Path
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
from label_ldm.label_generator.labelgen_data_utils import get_training_loaders
import mlflow.pytorch
# Filter Warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--data_dicts", type=str, help = "Path to the TSV containing paths to images")
    parser.add_argument("--data_dicts_val", type = str, help = "Path to the TSV containing paths to validation images")
    parser.add_argument("--checkpoints_dir", type=str, help = "Where do you want to save the models and logs.")
    parser.add_argument("--project_url", type=str, default="/project",
                        help="Path to the project where things will be saved.")
    parser.add_argument("--num_epochs", type = int, default = 100, help = "Number of epochs")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of threads")
    parser.add_argument("--validation_epochs", type = int, default=10, help="After how many epochs we validate")
    parser.add_argument("--save_every", type = int, default = 5, help = "Every how many epochs we save")
    parser.add_argument("--augmentation", type=int, default=0, help = "Use augmentation.")
    args = parser.parse_args()
    return args

def main(args):

    # Set mlflow
    set_determinism(42)

    # Set outputs directory (mlflow)
    #mlflow.set_tracking_uri(args.mlruns_path)
    output_dir = Path(f"{args.project_url}/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / args.checkpoints_dir
    # Load old model
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    # Print arguments.
    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    print("Loading configuration...")
    config = OmegaConf.load(args.config_file)

    # Check resolution: needs to be divisible by 2**(number downsamplings)
    new_res = checkResolution(config['stage1']['resolution'],
                              len(config['stage1']['params']['hparams']['num_channels']) - 2)
    if new_res != config['stage1']['resolution']:
        Warning("Changed resolution %s to  %s to make sure the downsamplings are fine. "
                % (str(config['stage1']['resolution']),
                   str(new_res)))
        config['stage1']['resolution'] = new_res

    print("Getting data...")

    train_loader, val_loader = get_training_loaders(
        batch_size=args.batch_size,
        training_ids=args.data_dicts,
        spatial_size=config['stage1']['resolution'],
        validation_ids=args.data_dicts_val,
        augmentation=bool(args.augmentation),
        num_workers=args.num_workers,
        conditionings=[],
        cache_dir=args.checkpoints_dir
    )

    print("Creating model...")

    model = AutoencoderKL(**config["stage1"]["params"]["hparams"])
    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="medicalnet_resnet10_23datasets",
                                     is_fake_3d=False, fake_3d_ratio=0.2)

    # Data Parallel
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)

    # Send to device.
    model = model.to(device)
    discriminator = discriminator.to(device)
    perceptual_loss = perceptual_loss.to(device)

    # Optimizers
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config["stage1"]["base_lr"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["stage1"]["disc_lr"])


    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
    else:
        print(f"No checkpoint found.")

    # Run training:

    # Define Losses (additional)
    focal_loss = FocalLoss(include_background=True, gamma = 3)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_referee = AdversarialReferee(n_steps=15, up_threshold=config['stage1']['adv_up_threshold'],
                                     down_threshold=config['stage1']['adv_down_threshold'],
                                     fake_filling_label = adv_loss.fake_label)


    def KL_loss(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]

    # Summary writers
    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    # Training
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_recon_epoch_loss_list = []
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    for epoch in range(start_epoch, args.num_epochs):
        print("epoch %d/%d" %(epoch, args.num_epochs))
        model.train().to(device)
        discriminator.train().to(device)
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        steps_trained_gen = 0
        steps_trained_dis = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=130)
        progress_bar.set_description(f"Epoch {epoch}")
        train_losses = {'recon_loss': 0, 'kld_loss': 0, 'perceptual_loss': 0, 'gen_loss': 0, 'dis_loss': 0}
        for step, batch in progress_bar:
            images = batch["label"].to(device)  # choose only one of Brats channels
            # Generator part
            optimizer_g.zero_grad(set_to_none=True)

            with autocast(enabled = True):
                reconstruction, z_mu, z_sigma = model(images)
                kl_loss = KL_loss(z_mu, z_sigma)
                recon_loss = focal_loss(reconstruction.float(), images.float())
                reconstruction = torch.softmax(reconstruction.float(), 1)
                p_loss = perceptual_loss(torch.argmax(reconstruction.float(), 1).unsqueeze(1).type(torch.float),
                                         torch.argmax(images.float(), 1).unsqueeze(1).type(torch.float)).mean()
                # Loss generator
                loss_g = recon_loss + config['stage1']['w_kl'] * kl_loss + config['stage1']['w_perceptual'] * p_loss

                if adv_referee.permission2trainGen():
                    steps_trained_gen += 1
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += config['stage1']['w_adversarial'] * generator_loss

            loss_g = loss_g.mean()
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # This needs to be computed
            logits_fake = discriminator(reconstruction.contiguous().detach().cpu())[-1]
            logits_real = discriminator(images.contiguous().detach().cpu())[-1]
            # Add this to referee
            adv_referee.calculate_and_add_accuracies(logits_real=logits_real, logits_fake=logits_fake,
                                                     )

            if adv_referee.permission2trainDis():
                # Discriminator part
                steps_trained_dis += 1
                optimizer_d.zero_grad(set_to_none=True)
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = config['stage1']['w_adversarial'] * discriminator_loss
                loss_d = loss_d.mean()
                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d = torch.tensor(0.0)

            epoch_loss += recon_loss.item()

            if adv_referee.permission2trainGen():
                gen_epoch_loss += generator_loss.item()
            if adv_referee.permission2trainDis():
                disc_epoch_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                        "accuracy_disc": adv_referee.getAccuracy()
                    }
                )

            train_losses['recon_loss'] += recon_loss.item()
            train_losses['kld_loss'] += kl_loss.item() * config['stage1']['w_kl']
            train_losses['perceptual_loss'] += p_loss.item() * config['stage1']['w_perceptual']
            train_losses['gen_loss'] += loss_g.item()
            train_losses['dis_loss'] += loss_d.item()

        # Global loss
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

        # Dictionary-based losses
        for key, val in train_losses.items():
            train_losses[key] = val / (step + 1)

        writer_train.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(train_loader) + step)
        writer_train.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(train_loader) + step)
        for k, v in train_losses.items():
            writer_train.add_scalar(f"{k}", v, epoch * len(train_loader) + step)

        # Validation
        if epoch % args.validation_epochs == 0:
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=110)
            progress_bar.set_description(f"Validation epoch {epoch}")
            val_losses = {'recon_loss': 0, 'kld_loss': 0, 'perceptual_loss': 0,
                          'gen_loss': 0, 'dis_loss': 0}
            to_plot_item = np.random.choice(len(val_loader)-1)
            for step, batch in progress_bar:
                images = batch["label"].to(device)  # choose only one of Brats channels
                with torch.no_grad():
                    with autocast(enabled=False):
                        reconstruction, z_mu, z_sigma = model(images)
                        kl_loss = KL_loss(z_mu, z_sigma)
                        recon_loss = focal_loss(reconstruction.float(), images.float())
                        reconstruction = torch.softmax(reconstruction.float(), 1)
                        p_loss = perceptual_loss(torch.argmax(reconstruction.float(), 1).unsqueeze(1).type(torch.float),
                                                 torch.argmax(images.float(), 1).unsqueeze(1).type(torch.float)).mean()
                        # Loss generator
                        loss_g = recon_loss + config['stage1']['w_kl'] * kl_loss + config['stage1']['w_perceptual'] * p_loss
                        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                        loss_g += config['stage1']['w_adversarial'] * generator_loss

                        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                        loss_d = config['stage1']['w_adversarial'] * discriminator_loss
                if step == to_plot_item:
                    reconstruction = reconstruction.detach().cpu() # We plot the reconstruction
                    gt = images.detach().cpu()
                    log_3d_img(reconstruction, None, gt, writer=writer_val, step=epoch,
                               n_plots=1)

                val_losses['recon_loss'] += recon_loss.item()
                val_losses['kld_loss'] += kl_loss.item() * config['stage1']['w_kl']
                val_losses['perceptual_loss'] = p_loss.item() * config['stage1']['w_perceptual']
                val_losses['gen_loss'] = loss_g.item()
                val_losses['dis_loss'] = loss_d.item()
                val_recon_epoch_loss_list.append(sum(val_losses.values()))

            for key, val in val_losses.items():
                val_losses[key] = val / len(val_loader)
            val_recon_epoch_loss_list[-1] /= len(val_loader)
            for k, v in train_losses.items():
                writer_val.add_scalar(f"{k}", v, epoch * len(val_loader) + step)

            print("Validation results: %s" %", ".join(["%s: %.6f" %(key, val) for key, val in val_losses.items()]))

        if epoch % args.save_every == 0 or epoch == args.num_epochs - 1:
            # Save model
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
            }

        # Save checkpoint
        torch.save(checkpoint, os.path.join(args.checkpoints_dir, "checkpoint.pth"))

        if val_recon_epoch_loss_list[-1] <= best_loss:
            print(f"New best val loss {val_recon_epoch_loss_list[-1]}")
            best_loss = val_recon_epoch_loss_list[-1]
            raw_model = model.module if hasattr(model, "module") else model
            torch.save(raw_model.state_dict(), str(os.path.join(args.checkpoints_dir, "best_model.pth")))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(model.state_dict(), os.path.join(args.checkpoints_dir,"final_model.pth"))
    print("Logging mlflow details...")
    if len(val_recon_epoch_loss_list) == 0:
        log_val_loss = best_loss
    else:
        log_val_loss = val_recon_epoch_loss_list[-1]
    raw_model = model.module if hasattr(model, "module") else model
    log_mlflow(
        model=raw_model,
        config=config,
        args=args,
        experiment="vae",
        run_dir=run_dir,
        val_loss=log_val_loss,
    )

    if os.path.isdir(os.path.join(args.checkpoints_dir, 'cache')):
        shutil.rmtree(os.path.join(args.checkpoints_dir, 'cache'))

args = parse_args()
main(args)