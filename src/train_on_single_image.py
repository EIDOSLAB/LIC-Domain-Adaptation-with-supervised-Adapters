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


from compress.entropy_models.weight_entropy_module import WeightEntropyModule

from compress.models import WACNNStanh, WACNN, QuantizedModelWrapper
from compress.ops.cdf import LogisticCDF, SpikeAndSlabCDF
from compress.models.utils import get_model
from compress.utils.image import read_image, pad, crop


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m","--model",default="cnn_base",choices=models.keys(),help="Model architecture (default: %(default)s)",)
    parser.add_argument("-d", "--dataset", type=str, default = "/scratch/dataset/openimages", help="Training dataset")
    parser.add_argument("-e","--epochs",default=100,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("--suffix",default=".pth.tar",type=str,help="factorized_annealing",)

    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument('--sgd', type = str,default = "sgd", help='use sgd as optimizer')


    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lmbda",type=float,default=0.0250,help="Bit-rate distortion parameter (default: %(default)s)",)


    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=64,help="Test batch size (default: %(default)s)",)
    parser.add_argument( "--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--save_path", type=str, default="7ckpt/model.pth.tar", help="Where to Save model")
    parser.add_argument("--seed", type=float,default = 42, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")


    parser.add_argument('--re_grad', '-rg', action='store_true', help='ste quantizer')


    parser.add_argument("-dims","--dimension",default=192,type=int,help="Number of epochs (default: %(default)s)",) 
    parser.add_argument("-dims_m","--dimension_m",default=320,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-q","--quality",default=3,type=int,help="Number of epochs (default: %(default)s)",)

    parser.add_argument("--distrib",type=str,choices={"spike-and-slab", "logistic"},default="logistic",)


    parser.add_argument("--fact_extrema",default=20,type=int,help="factorized_extrema",)
    parser.add_argument('--fact_tr', '-ft', action='store_true', help='factorized trainable')



    parser.add_argument("--gauss_extrema",default=120,type=int,help="gauss_extrema",)
    parser.add_argument('--gauss_tr', '-gt', action='store_true', help='gaussian trainable')


    parser.add_argument("--filename",default="/data/",type=str,help="factorized_annealing",)




    parser.add_argument( "--alpha", default=0.95, type=float, help="target mean squared error",)

    parser.add_argument( "--momentum", default=0.0, type=float, help="momnetum for the optimizer",)
    parser.add_argument( "--weight_decay", default=0.0, type=float, help="weigth dacay for the optimizer (L2)",)



    parser.add_argument( "--num_images_train", default=300000, type=int, help="images for training",)
    parser.add_argument( "--num_images_val", default=2048, type=int, help="images for validation",)  


    parser.add_argument("--pret_checkpoint",default = "/scratch/KD/devil2022/derivation_wa/no_mean/00670-devil2022-pth.tar") 
    #parser.add_argument("--pret_checkpoint",default = "/scratch/KD/devil2022/anchor/q5-zou22.pth.tar")



    parser.add_argument("--scheduler","-sch", type = str, default ="plateau")
    parser.add_argument("--patience",default=4,type=int,help="patience",)

    parser.add_argument("--trainable_lrp","-tlrp",action='store_true',)

    parser.add_argument('--training_policy', '-tp',type = str, default = "quantization", choices= ["entire_qe","quantization_lrp","residual","kd","entire", "quantization", "adapter","mse","controlled","only_lrp"] , help='adapter loss')

    parser.add_argument("--dim_adapter",default=1,type=int,help="dimension of the adapter",)
    parser.add_argument( "--mean", default=0.0, type=float, help="initialization mean",)
    parser.add_argument( "--std", default=0.02, type=float, help="initialization std",)
    parser.add_argument( "--kernel_size", default=1, type=int, help="initialization std",)


    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--width", type=float, default=0.06)
    parser.add_argument("--data_type", default="uint8")


    parser.add_argument("--regex", type=str, default="g_s\.5\.adapter.*")

    args = parser.parse_args(argv)
    return args


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
    bpp = sum(len(string[0]) for string in data["strings"]) * 8 / n_pixels

    return x_hat, psnr,bpp,mse


def prepare_model_with_adapters(model, args, device, w_ent):
    print("ora entro qua nel prepare model with adapters!!")
    # if n_adapters = 0 use ZeroLayer -- equivalent with no adapter

    state_dict = model.state_dict()

    if args.model == "latent":
        

        model.modify_adapter(args, device)
        model_qua = QuantizedModelWrapper(model, w_ent, regex=args.regex)
        # compute diff. from zero
        for key in model_qua.params_init.keys():
            if "adapter" in key:
                print("entro qui in linea 630")
                model_qua.params_init[key].fill_(0)

    else:
        assert (
            args.dim_adapter_1 == [0, 0, 0, 0, 0, 0, 0]
            and args.dim_adapter_2 is None
        )
        model_qua.regex = args.regex


    model_qua.report_params()
    model_qua.update_ent(force=True)
    model_qua.to(device)

    return model, model_qua



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

    """"
    x_hat, psnr, bpp, mse = evaluate(model, x,  actual=True)

    print("Before adapter otpimization <ACTUAL>  PSNR: {:.3f}, BPP: {:.4f}".format(psnr, bpp))

    transforms.ToPILImage()(x[0]).save(args.out / "input.png")
    transforms.ToPILImage()(x_hat[0]).save(args.out / "init.png") 

    # now the adapters have to be initialized 

    if args.dim_adapter != 0:
        model.modify_adapter(args, device) 
        model = model.to(device)


    if args.distrib == "spike-and-slab":
        distrib = SpikeAndSlabCDF(args.width, args.sigma, args.alpha)
    elif args.distrib == "logistic":
        distrib = LogisticCDF(scale=args.sigma)
    else:
        raise NotImplementedError

    w_ent = WeightEntropyModule(distrib, args.width, data_type=args.data_type)
    criterion = RateDistortionLoss(args.lmbda)


    model.to(device)
    w_ent.to(device)
    criterion.to(device)

    model, model_qua = prepare_model_with_adapters(model, args, device , w_ent)

    # now it is time to initialize the adapter and
    """ 
    



