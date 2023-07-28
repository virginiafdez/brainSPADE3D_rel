import os
import numpy as np
import matplotlib.pyplot as plt
from data_synthesis.brainspade_sampler import LabelSampler, ImageSampler
from omegaconf import OmegaConf
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type = str, help = "Path to yaml containing the label and image"
                                                            "generation configuration.")
    parser.add_argument("--batch_size", type=int, help="batch size")
    args = parser.parse_args()
    return args

def main(args):

    params =  OmegaConf.load(args.config_file)
    params['label_generator'].update(params['general'])
    params['image_generator'].update(params['general'])
    label_sampler = LabelSampler(**params['label_generator'])
    label_sampler.sampleLabel(args.batch_size)
    image_sampler = ImageSampler(**params['image_generator'])
    image_sampler.sampleImage(args.batch_size)

args = parse_args()
main(args)