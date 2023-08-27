import argparse

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
import sys

from compress.utils.help_function import CustomDataParallel, configure_latent_space_policy
from compress.training import train_one_epoch, test_epoch,  compress_with_ac, RateDistortionLoss , AdapterLoss, DistorsionLoss, RateDistortionModelLoss


from compress.entropy_models.weight_entropy_module import WeightEntropyModule

from compress.models import WACNNStanh, WACNN, QuantizedModelWrapper
from compress.ops.cdf import LogisticCDF, SpikeAndSlabCDF
from compress.models.utils import get_model, forward_pass
from compress.utils.image import read_image, pad, crop
from compress.zoo import models

CUDNN_INFERENCE_FLAGS = {"benchmark": False, "deterministic": True, "enabled": True}
torch.autograd.set_detect_anomaly(True)


def configure_optimizers(net, lr: float, aux_lr: float):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }


    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())



    optimizer = torch.optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )

    return optimizer




def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-p", "--path", type=str, default = "/scratch/dataset/kodak", help="Training dataset")

    parser.add_argument("-m","--model",default="latent",choices=models.keys(),help="Model architecture (default: %(default)s)",)
    parser.add_argument("-it","--iterations",default=10000,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("--suffix",default=".pth.tar",type=str,help="factorized_annealing",)

    parser.add_argument("-lr","--lr",default=1e-3,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument('--sgd', type = str,default = "sgd", help='use sgd as optimizer')
    parser.add_argument('--entropy_bot', '-eb', action='store_true', help='entropy bottleneck trainable')

    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lmbda",type=float,default=0.048350,help="Bit-rate distortion parameter (default: %(default)s)",)


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






    parser.add_argument("--pret_checkpoint",default = "/scratch/KD/devil2022/derivation_wa/no_mean/00050-devil2022-pth.tar") 
    #parser.add_argument("--pret_checkpoint",default = "/scratch/KD/devil2022/anchor/q5-zou22.pth.tar")



    parser.add_argument("--scheduler","-sch", type = str, default ="plateau")
    parser.add_argument("--patience",default=4,type=int,help="patience",)

    parser.add_argument("--trainable_lrp","-tlrp",action='store_true',)

    parser.add_argument('--training_policy', '-tp',type = str, default = "quantization", choices= ["entire_qe","quantization_lrp","residual","kd","entire", "quantization", "adapter","mse","controlled","only_lrp"] , help='adapter loss')

    parser.add_argument("--dim_adapter",default=0,type=int,help="dimension of the adapter",)
    parser.add_argument( "--mean", default=0.0, type=float, help="initialization mean",)
    parser.add_argument( "--std", default=0.00, type=float, help="initialization std",)
    parser.add_argument( "--kernel_size", default=1, type=int, help="initialization std",)


    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--width", type=float, default=0.06)
    parser.add_argument("--data_type", default="uint8")


    parser.add_argument("--regex", type=str, default="adapter.*")
    parser.add_argument("--out", type=Path, default="../temp/")

    args = parser.parse_args(argv)
    return args




def optimize_adapter(
    model_qua: QuantizedModelWrapper,
    model: nn.Module,
    criterion: RateDistortionLoss,
    x: torch.Tensor,
    iterations: int,
    lr: float,
    y: torch.Tensor,
    z: torch.Tensor,
) -> None:
    model_qua.eval()
    model_qua.w_ent.train()


    model.pars_adapter( re_grad = False)
    model_qua.model.pars_adapter( re_grad = False)

    model.unfreeze_quantizer()
    model_qua.model.unfreeze_quantizer()

    x_ = pad(x)
    # Calcolo del numero di parametri
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad )
    print(f"Numero totale di parametri allenabili model qua: {num_params}")

    # Calcolo del numero di parametri
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad )
    print(f"Numero totale di parametri allenabili model: {num_params}")


    

    optimizer  = configure_optimizers(model, lr, 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(iterations * 4 // 10), gamma=0.3)
    start = time.time()
    for it in range(iterations):


        optimizer.zero_grad()
        # model weights -> model_qua weights -> loss
        m_likelihoods = model_qua.update_parameters(model)

        output = forward_pass(model, x_) 






        output["x_hat"] = crop(output["x_hat"], x.shape[2:])
        output["m_likelihoods"] = m_likelihoods
        out_criterion: dict = criterion(output, x_)


        
        if it==0: 
            mse: torch.Tensor = (x - output["x_hat"]).square().mean(dim=(1, 2, 3))
            psnr: torch.Tensor = -10 * mse.log10().mean() 
            print("at first iterations we have ",psnr)          

        out_criterion["loss"].backward(retain_graph = True)
        optimizer.step()
        lr_scheduler.step()
        mse: torch.Tensor = (x - output["x_hat"]).square().mean(dim=(1, 2, 3))
        psnr: torch.Tensor = -10 * mse.log10().mean()

        if it%100==0:
            print("Loss: {:.4f}, Time: {:.2f}s, lr: {}, it: {}".format(
                out_criterion["loss"].item(),
                time.time() - start,
                optimizer.param_groups[0]["lr"],
                it
            ))
            print("--- ",psnr,"----", out_criterion["bpp_loss"] )

        #logging.info(
        #    "Loss: {:.4f}, Time: {:.2f}s, lr: {}".format(
        #        out_criterion["loss"].item(),
        #        time.time() - start,
        #        optimizer.param_groups[0]["lr"],
        #    )
        #)

    # final update
    with torch.no_grad(), torch.backends.cudnn.flags(**CUDNN_INFERENCE_FLAGS):
        model_qua.update_parameters(model)


def evaluate(model,x):

    height, width = x.shape[2:]
    n_pixels: int = height * width
    x_ = pad(x)


    data =  model.compress(x_.detach())
    
    out_dec = model.decompress(data["strings"], data["shape"])

    
    x_hat = crop(out_dec["x_hat"], x.shape[2:])
    x_hat = x_hat.mul(255).round().div(255)




    mse: torch.Tensor = (x - x_hat).square().mean(dim=(1, 2, 3))
    psnr: torch.Tensor = -10 * mse.log10().mean()
    bpp = 0 #sum(len(string[0]) for string in data["strings"]) * 8 / n_pixels

    return x_hat, psnr , bpp, mse,data["z"].detach(), data["y"].detach()


def prepare_model_with_adapters(model, args, device, w_ent):

    # if n_adapters = 0 use ZeroLayer -- equivalent with no adapter

    state_dict = model.state_dict()



    if args.model == "latent":
        

        
        model_qua = QuantizedModelWrapper(model, w_ent, regex=args.regex)
        # compute diff. from zero
        for key in model_qua.params_init.keys():
            if "adapter" in key:
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

import os

def main(argv):
    args = parse_args(argv)
    device =  "cuda"
    if args.model == "base":
        factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = True)
    else:
        factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = False)
    
    lista_immagini = [os.path.join(args.path,f) for f in os.listdir(args.path)]
    pth = lista_immagini[1]
    nome_immagine = pth.split(".")[0]
    img: Image.Image = read_image(pth)

    
    #logging.basicConfig(filename=(args.out / "log"), filemode="w", level=logging.INFO, format=fmt)
    transform = transforms.ToTensor()
    x: torch.Tensor = transform(img)[None]  # .repeat(16, 1, 1, 1)
    x = x.to(device)
    x_pad = pad(x)
    print("x.shape",x_pad.shape)

    model, baseline = get_model(args,device, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration )



    lista_chiavi = list(model.state_dict().keys())
    """    for i,l in enumerate(lista_chiavi):
        if "adapter" in l:
            print(model.state_dict()[l]," ",l)"""





    ##### evaluate the first time without adapter! 
    model.freeze_net()
    # Calcolo del numero di parametri
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad )
    print(f"NUMERO INIZIALE PARAMETRI ALLENABILI: {num_params}")
    


    if True: #"y.pth" not in os.listdir(args.out / "files"):
        x_hat, psnr, bpp, mse, z, y = evaluate(model, x)

        print("Before adapter otpimization <ACTUAL>  PSNR: {:.3f}, BPP: {:.4f}".format(psnr, bpp))
    
        transforms.ToPILImage()(x[0]).save(args.out / "images/input.png")
        transforms.ToPILImage()(x_hat[0]).save(args.out / "images/init.png") 


        # Salva il tensore nel file specificato

        torch.save(z,args.out / "files/z_hat.pth" )
        torch.save(y,args.out / "files/y.pth" )
    else: 
        z = torch.load(args.out / "files/z_hat.pth" )
        y = torch.load(args.out / "files/y.pth" )


    # now the adapters have to be initialized 


    if args.distrib == "spike-and-slab":
        distrib = SpikeAndSlabCDF(args.width, args.sigma, args.alpha)
    elif args.distrib == "logistic":
        distrib = LogisticCDF(scale=args.sigma)
    else:
        raise NotImplementedError

    w_ent = WeightEntropyModule(distrib, args.width, data_type=args.data_type)
    criterion =RateDistortionLoss(args.lmbda)


    model.to(device)
    w_ent.to(device)
    criterion.to(device)


    if args.dim_adapter != 0:
        model.modify_adapter(args, device) 
        model = model.to(device)


    model, model_qua = prepare_model_with_adapters(model, args, device , w_ent)

    # now it is time to initialize the adapter and
    
    # encoder and entropy models are in evaluation mode.
    
    model_qua.eval_enc()
    #model_qua.train()
    model_qua.freeze_net()
    model_qua.train_adapter()


    
    
    # Calcolo del numero di parametri
    num_params = sum(p.numel() for p in model_qua.model.parameters() if p.requires_grad )
    print(f"Numero totale di parametri nel layer nn.Conv2d: {num_params}")

    

    # RIPARTIRE DA QUA!!!!
    optimize_adapter(model_qua,model,criterion,x,args.iterations,args.lr,y=y,z = z )

if __name__ == "__main__":
    main(sys.argv[1:])