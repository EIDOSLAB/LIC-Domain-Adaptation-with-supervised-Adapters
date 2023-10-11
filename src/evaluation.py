import torch 
import os 
import numpy as np 

from torchvision import transforms
from PIL import Image
import torch
from compress.datasets import AdapterDataset

import wandb

from compress.training.step_gate import  compress_with_ac_gate
import numpy as np
import sys
from compress.zoo import models

from compress.utils.parser import parse_args_evaluation
from compress.training import compress_with_ac


def rename_key_for_adapter(key, stringa, nuova_stringa):
    if key.startswith(stringa):
        key = nuova_stringa # nuova_stringa  + key[6:]
    return key


def from_state_dict(cls, state_dict):
    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net



def get_model_for_evaluation(model, pret_checkpoint, device):



    if True: #args.name_model == "WACNN":
        checkpoint = torch.load(pret_checkpoint , map_location=device)
        if model == "base":
            checkpoint = torch.load(pret_checkpoint , map_location=device)#["state_dict"]

            net = from_state_dict(models[model], checkpoint)
            net.update()
            net.to(device) 
            return net
        else:

            print("-----------------------------> ",list(checkpoint.keys()))
            state_dict = checkpoint["state_dict"]
            args = checkpoint["args"]



            print(args)

            print("----> ",models)


            net = models[model](N = args.N,
                                M =args.M,
                                dim_adapter_attn = args.dim_adapter_attn,
                                stride_attn = args.stride_attn,
                                kernel_size_attn = args.kernel_size_attn,
                                padding_attn = args.padding_attn,
                                type_adapter_attn = args.type_adapter_attn,
                                position_attn = args.position_attn,
                                num_adapter = args.num_adapter,
                                aggregation = args.aggregation,
                                std = args.std,
                                mean = args.mean,                              
                                bias = args.bias,
                                skipped = False
                                    ) 
   
        info = net.load_state_dict(state_dict, strict=False)
        net.to(device)
        return net





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






def read_image(filepath):
    #assert filepath.is_file()
    img = Image.open(filepath)
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

    

    device = "cuda" if  torch.cuda.is_available() else "cpu" #dddd



    root = "/scratch/dataset/DomainNet/splitting/mixed"
    #path_models = "/scratch/KD/devil2022/adapter/3_classes/q5"
    path_models = "/scratch/universal-dic/weights/q5"

    test_transforms = transforms.Compose([transforms.ToTensor()])
    print("****************************************** PAINTING***************************************+++*************************")
    painting = AdapterDataset(root = root, path  =  ["test_infograph.txt"], transform = test_transforms,train = False)
    painting_f = painting.samples
    painting_filelist = []

    for i in range(len(painting_f)):
        painting_filelist.append(painting_f[i][0])





    







    list_models = [os.path.join(path_models,f) for f in os.listdir(path_models)]
    mode = "base"
    for m in list_models:

        net = get_model_for_evaluation(mode,m,device)
        if mode == "gate":
            psnr, bpp = compress_with_ac_gate(net, 
                                        painting_filelist, 
                                        device, 
                                        -1, 
                                        3,
                                        loop=False, 
                                        name =  "infograph_", 
                                        oracles= None, 
                                        train_baseline= False,
                                        writing = None)
        else:
            psnr,bpp =  compress_with_ac(net, painting_filelist, device, -1, loop=False)
            print(psnr,"  ",bpp)






    
if __name__ == "__main__":

    wandb.init(project="prova", entity="albertopresta")   
    main(sys.argv[1:])
