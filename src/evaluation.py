import torch 
import os 
import numpy as np 

from torchvision import transforms
from PIL import Image
import torch
import pickle

import wandb


import numpy as np
import sys


from compressai.zoo import *


from compress.utils.parser import parse_args_evaluation
from compress.models.utils import get_model_for_evaluation

from compress.training import compress_with_ac











IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)






def read_image(filepath, clic =False):
    #assert filepath.is_file()
    img = Image.open(filepath)
    
    if clic:
        i =  img.size
        i = i[0]//2, i[1]//2
        img = img.resize(i)
    img = img.convert("RGB")
    return transforms.ToTensor()(img) 





"""
def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics
"""

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def main(argv):
    args =  parse_args_evaluation(argv)
    set_seed(seed = args.seed)

    print(args,"cc")

    

    device = "cuda" if  torch.cuda.is_available() else "cpu"

    if args.test_dataset in ("kodak","clic"):

        pth = os.path.join("/scratch/dataset/",args.test_dataset)
        filelist = [os.path.join(pth,f) for f in os.listdir(pth)]
    else:
        path = os.path.join("/scratch/dataset/PACS/splitting", args.test_dataset,str(args.seed),"file.pkl")

        filelist = pickle.load(path)["test"]

        






    




    #if args.model == "base":

    #    factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = True)
    #else:
    #    factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = False)
    #print("gaussian configuration----- -fdddguuggffxssssxxx------>: ",gaussian_configuration)
    #print("factorized configuration------>ceeccccssààcccc->: ",factorized_configuration)



    list_models = [os.path.join(args.path_models,f) for f in os.listdir(args.path_models)]

    for m in list_models:
        if args.targ_q in m or args.targ_q == "all":
            net, baseline = get_model_for_evaluation(args,m,device)
            net = net.to(device)
            sos = not baseline

            with torch.no_grad():
                pth = "/scratch/KD/devil2022/results/derivation"
                writing_seq = m.split("/")[-1].split(".")[0]

                b = compress_with_ac(net, filelist, device, -1, loop=False, writing = os.path.join(pth,writing_seq))

        else:
            print("questo modello non è nel target: ",m)





    
if __name__ == "__main__":

    wandb.init(project="prova", entity="albertopresta")   
    main(sys.argv[1:])
