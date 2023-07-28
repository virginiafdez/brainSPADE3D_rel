import monai
import torch
import numpy as np
import os
from tensorboardX import SummaryWriter
from pathlib import PosixPath
import mlflow.pytorch
import numpy as np
import torch
from mlflow import start_run
from omegaconf import OmegaConf, DictConfig
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR, \
    CosineAnnealingWarmRestarts, ConstantLR, LambdaLR
from copy import deepcopy


def checkResolution(resolution, n_downsamplings):

    out_res = []
    for r in resolution:
        if r % (2**n_downsamplings) != 0:
            j = r-1
            found= False
            while not found:
                remaining = j % (2**n_downsamplings)
                if remaining == 0:
                    found = True
                    out_res.append(j)
                j = j-1
        else:
            out_res.append(r)
    return out_res

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

def log_3d_img(
        rec_imgs: torch.Tensor,
        cond_list: torch.Tensor,
        gt_imgs: torch.Tensor,
        writer: SummaryWriter,
        step: int,
        n_plots = 1,
        conditionings = None,
        is_train = False,
):

    # Plot images
    if n_plots > rec_imgs.shape[0]:
        n_plots = rec_imgs.shape[0]

    # Tensor should be BxCxHxWxD
    rec_imgs = rec_imgs.detach().cpu()
    rec_img = torch.argmax(rec_imgs, 1).numpy() # B x H x W x D
    if gt_imgs is not None:
        gt_imgs = gt_imgs.detach().cpu()
        gt_img = torch.argmax(gt_imgs, 1).numpy() # B x H x W x D

    img_ids = np.random.choice(range(rec_img.shape[0]), n_plots, replace = False)
    img_ids = [int(i) for i in img_ids]

    for i in img_ids:
        title = ""
        if conditionings is not None and cond_list is not None:
            cond_i = cond_list[i, ...]
            for l_ind,l in enumerate(conditionings):
                if cond_i[l_ind] == 1:
                    title += "-%s" %l

        # rec_img_ = colors[rec_img[i, ...]].transpose(-1, 0, 1, 2) # H x W x D x 3 (color) > 3 x H x W x D
        # gt_img_ = colors[gt_img[i, ...]].transpose(-1, 0, 1, 2) # H x W x D x 3 (color) > 3 x H x W x D
        rec_img_ = np.expand_dims(rec_img[i, ...], 0) # 1 x H x W x D
        if gt_imgs is not None:
            gt_img_ = np.expand_dims(gt_img[i, ...], 0)  # 1 x H x W x D
            out_img = np.concatenate([rec_img_, gt_img_], 1)
        else:
            out_img = rec_img_

        individual_channels = []
        individual_channels_gt = []
        for ch in range(rec_imgs.shape[1]):
            individual_channels.append(np.expand_dims((rec_imgs[i, ch, ...]*8).numpy(), 0))
            if gt_imgs is not None:
                individual_channels_gt.append(np.expand_dims((gt_imgs[i, ch, ...] * 8).numpy(),0))
            # Each time is 3 x H x W x D
        individual_channels = np.concatenate(individual_channels, 2)
        if gt_imgs is not None:
            individual_channels_gt = np.concatenate(individual_channels_gt, 2)
            individual_channels = np.concatenate([individual_channels, individual_channels_gt], 1)

        out_img = np.concatenate([out_img, individual_channels], 2)
        out_img = np.expand_dims(out_img, 0)
        #individual_channels = np.expand_dims(individual_channels, 0)
        if is_train:
            tag = "train_ims/%s_%d" %(title,i)
        else:
            tag = "val_ims/%s_%d" % (title, i)
        monai.visualize.plot_2d_or_3d_image(data=out_img, step=step, writer=writer,
                                            index=0,
                                            tag=tag, max_channels=1,
                                            max_frames=6, frame_dim=-1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def recursive_items(dictionary, prefix = ""):
    for key, value in dictionary.items():
        if type(value) in [dict, DictConfig]:
            yield from recursive_items(value, prefix=str(key) if prefix == "" else f"{prefix}.{str(key)}")
        else:
            yield (str(key) if prefix == "" else f"{prefix}.{str(key)}", value)


def log_mlflow(
        model,
        config,
        args,
        experiment: str,
        run_dir: PosixPath,
        val_loss: float,
):

    """Log model and performance on Mlflow system"""

    config = {
        **OmegaConf.to_container(config),
        **vars(args)
    }

    mlflow.set_tracking_uri("file:/ws_virginia/allcode/brainSPADE3D/mlruns")
    print(f"Setting mlflow experiment: {experiment}")
    mlflow.set_experiment(experiment)

    with start_run():
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in recursive_items(config):
            mlflow.log_param(key, value)

        mlflow.log_artifacts(str(run_dir / 'train'), artifact_path="events_train") #
        mlflow.log_artifacts(str(run_dir / 'val'), artifact_path="events_val")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model

        mlflow.pytorch.log_model(raw_model, "final_model")
        raw_model.load_state_dict(torch.load(str(run_dir / "best_model.pth")), strict = False)
        mlflow.pytorch.log_model(raw_model, "best_model")


class AdversarialReferee:

    def __init__(self, n_steps, up_threshold, down_threshold, fake_filling_label):

        self.n_steps = n_steps
        self.accuracy_vector = []
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.fake_filling_label = 0.0

    def calculate_accuracies(self, logits, ground_truth):
        accuracy = torch.abs(logits-ground_truth).mean()
        return accuracy

    def calculate_and_add_accuracies(self, logits_real, logits_fake):
        ground_truth_real = self.get_target_tensor(logits_real, target_is_real=True)
        ground_truth_fake = self.get_target_tensor(logits_fake, target_is_real=False)
        accuracy_reals = torch.abs(1-(ground_truth_real - torch.sigmoid(logits_real))).mean().detach().cpu().numpy()
        accuracy_fakes = torch.abs(1-(torch.sigmoid(logits_fake) - ground_truth_fake)).mean().detach().cpu().numpy()
        accuracy = (accuracy_reals + accuracy_fakes) / 2
        if len(self.accuracy_vector) < self.n_steps:
            self.accuracy_vector.append(accuracy)
        else:
            self.accuracy_vector =  self.accuracy_vector[1:] + [accuracy]

    def permission2trainGen(self):

        if len(self.accuracy_vector) == 0:
            return True
        return np.mean(self.accuracy_vector) >= self.down_threshold

    def permission2trainDis(self):

        if len(self.accuracy_vector) == 0:
            return True
        return np.mean(self.accuracy_vector) <= self.up_threshold

    def getAccuracy(self):

        if len(self.accuracy_vector) == 0:
            return 0.0
        else:
            return  np.mean(self.accuracy_vector)


    def get_target_tensor(self, input: torch.FloatTensor, target_is_real: bool) -> torch.Tensor:
        """
        Gets the ground truth tensor for the discriminator depending on whether the input is real or fake.

        Args:
            input: input tensor from the discriminator (output of discriminator, or output of one of the multi-scale
            discriminator). This is used to match the shape.
            target_is_real: whether the input is real or wannabe-real (1s) or fake (0s).
        Returns:
        """

        filling_label = 1.0 if target_is_real else self.fake_filling_label
        label_tensor = torch.tensor(1).fill_(filling_label).type(input.type())
        label_tensor.requires_grad_(False)
        return label_tensor.expand_as(input)

class TimeStepManager:

    def __init__(self, scheduler, num_train_timesteps, use_loss_buffer = False, loss_buffer_interval = 50,
                 use_loss_buffer_interval_from = 5) -> None:

        # Loss buffer
        self.num_train_timesteps = num_train_timesteps
        self.buffer_interval = loss_buffer_interval
        self.buffer_decay = 0.99
        self.loss_buffer = [0]*int(self.num_train_timesteps/self.buffer_interval) # Stores avg loss per interval
        self.use_loss_buffer = use_loss_buffer
        self.loss_reporting_tool = [0] * self.num_train_timesteps # Stores avg loss per time step
        self.use_loss_buffer_interval_from = use_loss_buffer_interval_from
        self.scheduler = scheduler

    def selectTimeSteps(self, epoch, batch_size, device):
        if epoch < self.use_loss_buffer_interval_from:
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps,(batch_size,), device=device).long()
        else:
            # Highest loss
            timesteps = torch.randint(low=(np.argmax(self.loss_buffer))*self.buffer_interval,
                                      high=(np.argmax(self.loss_buffer)+1)*self.buffer_interval,
                                      size= (batch_size, ),
                                      device=device).long()
        return timesteps

    def manageLossBuffer(
            self,
            loss: torch.Tensor,
            timesteps: torch.Tensor,
        ) -> None:


        for b in range(loss.shape[0]):
            self.loss_reporting_tool[timesteps[b]] = self.loss_reporting_tool[timesteps[b]] * self.buffer_decay + \
                                             (1 - self.buffer_decay) * loss[b,].item()

        self.loss_buffer[int(timesteps[0] // self.buffer_interval)] = self.buffer_decay * \
                                                              self.loss_buffer[int(timesteps[0] // self.buffer_interval)] + \
                                                              (1 - self.buffer_decay) * loss.mean().item()


def findLRGamma(startingLR, endingLR, num_epochs):
    '''
    Gamma getter. Based on a minimum and maximum learning rate, calculates the Gamma
    necessary to go from minimum to maimum in num_epochs.
    :param startingLR: First Learning Rate.
    :param endingLR: Final Learning Rate.
    :param num_epochs: Number of epochs.
    :return:
    '''

    gamma = np.e ** (np.log(endingLR / startingLR) / num_epochs)
    return gamma

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def getLearningRate(optimizer, warmup_lr, base_lr, n_epochs_warmup,
                    n_epochs_shift, n_epochs, lr_scheduler_type='linear',
                    end_lr=0.0000001, sanity_test = True):

    if n_epochs_shift > n_epochs:
        ValueError("Argument n_epochs cannot be smaller than n_epochs_shift!")

    lr_scheduler_type = lr_scheduler_type.lower()
    gamma_shift = findLRGamma(warmup_lr, base_lr, n_epochs_shift-n_epochs_warmup)

    if lr_scheduler_type == 'constant':
        lambda1 = lambda epoch: int(0 <= epoch < n_epochs_warmup) * (1.0) \
                                + int(n_epochs_warmup <= epoch < n_epochs_shift) * (
                                            gamma_shift ** (epoch - n_epochs_warmup - 1)) \
                                + int(n_epochs_shift <= epoch) * (1 / base_lr) * 1.0

    elif lr_scheduler_type == 'exponential':
        # lambda1 = lambda epoch: int(0 <= epoch < 10) * (1.0) + int(10 <= epoch < 25) * (
        #             findLRGamma(0.00000001, 0.0001, 15) ** (epoch - 9)) + int(25 <= epoch) * (base_lr/warmup_lr) * (
        #                                     findLRGamma(0.0001, 0.000001, 275) ** (epoch - 24))

        gamma_decay = findLRGamma(base_lr, end_lr, n_epochs - n_epochs_shift)
        lambda1 = lambda epoch: int(0 <= epoch < n_epochs_warmup) * (1.0) \
                                + int(n_epochs_warmup <= epoch < n_epochs_shift) * (gamma_shift ** (epoch - n_epochs_warmup -1)) \
                                + int(n_epochs_shift <= epoch) * (base_lr/warmup_lr) * gamma_decay ** (epoch - n_epochs_shift - 1)

    else:
        ValueError("Learning rate type not supported. Enter: constant, linear, cosine or exponential."
                   "%s was found" %lr_scheduler_type)

    scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])

    # Before the scheduler, we run a test. We need to make sure it doesn't NaN because of Overflows:
    dummy_net = torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3) # Dummy
    dummy_optimizer = torch.optim.Adam(dummy_net.parameters(), lr=warmup_lr) # Dummy
    dummy_scheduler = LambdaLR(dummy_optimizer, lr_lambda=[lambda1])
    for e in range(n_epochs):
        dummy_optimizer.step()
        dummy_scheduler.step()
        if np.isnan(get_lr(optimizer)):
            ValueError("Current learning rate settings will cause the error to be NaN."
                       "This is because the difference between the number of epochs and the warmup LR"
                       "is too big. Change it. If you are extending the number of epochs, "
                       "just increase the warmup value. It doesn't matter anyway.")
    return scheduler

def pad_latent(latent_space_shape, n_downsamplings):
    '''
    Finds the new latent space shape making possible to train the unet
    '''

    divisor = 2**n_downsamplings
    adjust = False
    for i in latent_space_shape[1:]:
        if i%divisor != 0:
            adjust = True
            break
    if adjust:
        new_shape = []
        for i in latent_space_shape[1:]:
            floating_i = i
            ok = False
            while not ok:
                if floating_i%divisor != 0:
                    floating_i += 1

                else:
                    new_shape.append(floating_i)
                    ok = True
        return True, [latent_space_shape[0]] + new_shape
    else:
        return False, list(latent_space_shape)
