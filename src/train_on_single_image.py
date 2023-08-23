import argparse
import copy
import logging
import operator
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms


from compress.utils.help_function import CustomDataParallel, configure_optimizers, configure_latent_space_policy
from compress.training import train_one_epoch, test_epoch,  compress_with_ac, RateDistortionLoss , AdapterLoss, DistorsionLoss

from compress.models import WACNNStanh, WACNN
from compress.models.utils import get_model
from compress.utils.image import read_image, pad, crop



def evaluate(model,x):

    height, width = x.shape[2:]
    n_pixels: int = height * width
    x_ = pad(x)


    data =  model.compress(x_)
    out_dec = model.decompress(data["strings"], data["shape"])

    x_hat = crop(out_dec["x_hat"], x.shape[2:])
    x_hat = x_hat.mul(255).round().div(255)




    mse: torch.Tensor = (x - x_hat).square().mean(dim=(1, 2, 3))
    psnr: torch.Tensor = -10 * mse.log10().mean()
    bpp = sum(len(string[0]) for string in c["strings"]) * 8 / n_pixels

    return x_hat, psnr,bpp,mse



def main(args):

    device = "cpu" if args.no_cuda else "cuda"
    if args.model == "cnn_base":
        factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = True)
    else:
        factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = False)
    

    img: Image.Image = read_image(args.path)

    transform = transforms.ToTensor()
    x: torch.Tensor = transform(img)[None]  # .repeat(16, 1, 1, 1)
    x = x.to(device)
    x_pad = pad(x)

    model, baseline = get_model(args.model,  args.model_path, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration )


    ##### evaluate the first time without adapter! 
    x_hat, psnr, bpp, mse = evaluate(model, x,  actual=True)

    print("Before adapter otpimization <ACTUAL>  PSNR: {:.3f}, BPP: {:.4f}".format(psnr, bpp))

    transforms.ToPILImage()(x[0]).save(args.out / "input.png")
    transforms.ToPILImage()(x_hat[0]).save(args.out / "init.png") 

    if args.dim_adapter != 0:
        model.modify_adapter(args, device) 
        model = model.to(device)


    # now it is time to initialize the adapter and 
    



