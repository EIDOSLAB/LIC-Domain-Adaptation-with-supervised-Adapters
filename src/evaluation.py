import torch 
import os 
import numpy as np 
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import math
from compressai.ops import compute_padding
import math 
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import numpy as np
import sys

import argparse
from compressai.zoo import *
from torch.utils.data import DataLoader
from os.path import join 
from compress.zoo import models, aux_net_models
import wandb
from torch.utils.data import Dataset
from os import listdir
from compress.utils.parser import parse_args_evaluation









class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

image_models = {
                "devil2022":models["stanh_cnn"],
                "basedevil2022":models["cnn_base"]
                }






def rename_key(key):
    """Rename state_deeict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]
    if key.startswith('h_s.'):
        return None

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    # if ".downsample." in key:
    #     return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key



def load_state_dict(state_dict):
    """Convert state_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict

def load_checkpoint(arch: str, checkpoint_path: str):
    state_dict = load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return models[arch].from_state_dict(state_dict).eval()




def bpp_calculation(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]

        bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
        #print("la lunghezza è: ",len(out_enc[1]))
        bpp_2 =  sum( (len(out_enc[1][i]) * 8.0 ) / num_pixels for i in range(len(out_enc[1])))
        return bpp_1 + bpp_2, bpp_1, bpp_2


@torch.no_grad()
def inference(model, filelist, device, model_name, entropy_estimation = False):
    # tolgo il dataloader al momento
    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    bpps = AverageMeter()
    quality_level =model_name.split("-")[0]
    print("inizio inferenza -----> ",quality_level)
    i = 0
    for d in filelist:
        name = "image_" + str(i)
        i +=1
        x = read_image(d).to(device)
        x = x.unsqueeze(0) 
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        x_padded = F.pad(x, pad, mode="constant", value=0)

        if entropy_estimation is False:
            data =  model.compress(x_padded)
            out_dec = model.decompress(data["strings"], data["shape"])
    
        else:
            out_dec = model(x_padded, training = False)

        if entropy_estimation is False:
            #print("SONO QUAAAAAA")
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            #print("lo shape decoded è-------------------------------> ",out_dec["x_hat"].shape," ",x.shape)
            out_dec["x_hat"].clamp_(0.,1.)
            metrics = compute_metrics(x, out_dec["x_hat"], 255)
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]

            bpp = sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels

            #bpp_z = len(data["strings"][0][0])*8.0/num_pixels
            #bpp_y = len(data["strings"][1][0])*8.0/num_pixels
            #print("i bit per pixels sono: ",bpp_y,"  ",bpp_z,"  ",bpp)


        else:
            out_dec["x_hat"].clamp_(0.,1.)
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]
            bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_dec["likelihoods"].values())
            metrics = compute_metrics(x, out_dec["x_hat"], 255)
        
        
        if i <= -1:

            folder_path = "/scratch/KD/devil2022/images/kodak/" + quality_level
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Cartella '{folder_path}' creata.")
                #else:
                #    print(f"La cartella '{folder_path}' esiste già.")
            image = transforms.ToPILImage()(out_dec['x_hat'].squeeze())
            nome_salv = os.path.join(folder_path, name + ".png")#"/scratch/inference/images/devil2022/3anchors" + name +    ".png"
            image.save(nome_salv)

        

        psnr.update(metrics["psnr"])
        if i%2==0:
            print(name,": ",metrics["psnr"]," ",bpp, metrics["ms-ssim"])
        ms_ssim.update(metrics["ms-ssim"])
        #bpps.update(bpp.item())
        bpps.update(bpp)

     
    print("fine indferenza per il seguente modello ",quality_level,": ",psnr.avg, ms_ssim.avg, bpps.avg)
    return psnr.avg, ms_ssim.avg, bpps.avg





@torch.no_grad()
def eval_models(res, dataloader, device, entropy_estimation):
  
    metrics = {}
    models_name = list(res.keys())
    for i, name in enumerate(models_name): #name = q1-bmshj2018-base/fact
        #print("----")
        print("name: ",name)

        model = res[name]["model"]


        psnr, mssim, bpp =  inference(model,dataloader,device,  name, entropy_estimation= entropy_estimation)

        metrics[name] = {"bpp": bpp,
                        "mssim": mssim,
                        "psnr": psnr
                            } 

       
    return metrics   













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


def load_pretrained(state_dict):
    """Convert sccctaddte_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict



def load_models(dict_model_list,  models_path, device, image_models):

    res = {}
    for i, name in enumerate(list(dict_model_list.keys())): #dict_model_listload
        nm = name.split("-")[1] # nome del modello da caricare
        
        print("--> ",dict_model_list[name])

        print("vedo che tipo di nome è venuto  ",nm,"<-------- ")            
        architecture =  image_models[nm]
        pt = os.path.join(models_path, dict_model_list[name])
        checkpoint = torch.load(pt, map_location=device)
        state_dict = checkpoint['state_dict']


        del state_dict["gaussian_conditional._offset"] 
        del state_dict["gaussian_conditional._quantized_cdf"] 
        del state_dict["gaussian_conditional._cdf_length"] 
        del state_dict["gaussian_conditional.scale_table"] 

        del state_dict["entropy_bottleneck._quantized_cdf"]
        del state_dict["entropy_bottleneck._cdf_length"]


        del state_dict["entropy_bottleneck._offset"]

        



        N = 192
        M = 320

        #factorized_configuration =checkpoint["factorized_configuration"]
        if "gaussian_configuration" in list(checkpoint.keys()):
            gaussian_configuration =  checkpoint["gaussian_configuration"]

            model =architecture(N = N, M = M, gaussian_configuration = gaussian_configuration)
        else:
            if "nolrp" in pt:
                trp = False
            else:
                trp = True 
            model =architecture(N = N, M = M, trainable_lrp = trp)

        model = model.to(device)     
        #model.update( device = device)
        model.load_state_dict(state_dict)  
        if  "factorized_configuration" in list(checkpoint.keys()):
            model.entropy_bottleneck.sos.update_state(device = device )


        if "gaussian_configuration" in list(checkpoint.keys()):    
            model.gaussian_conditional.sos.update_state(device = device)
            print("I HAVE DONE THE UPDATE!!!!-----> ",model.gaussian_conditional.sos.cum_w)
        
        
        model.update( force = True)
        
        res[name] = { "model": model}

        print(" il valore di trainable lrp è il seguente: ",model.trainable_lrp)
        #print("HO APPENA FINITO DI CARICARE I MODELLI")
    return res


def read_image(filepath, clic =False):
    #assert filepath.is_file()
    img = Image.open(filepath)
    
    if clic:
        i =  img.size
        i = i[0]//2, i[1]//2
        img = img.resize(i)
    img = img.convert("RGB")
    return transforms.ToTensor()(img) 






def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def main(argv):
    set_seed(seed = 42)
    args =  parse_args_evaluation(argv)
    model_name = args.model  # nome del modello che voglio studiare (ad esempio cheng2020)



    if args.lrp is False:
        models_path = join(args.model_path,model_name) # percorso completo per arrivare ai modelli salvati (/scratch/inference/pretrained_models/chegn2020) qua ho salvato i modelli 
    else:
        models_path = os.path.join(args.model_path,args.lrp_path,args.gamma)
 

    models_checkpoint = listdir(models_path) # checkpoints dei modelli  q1-bmshj2018-sos.pth.tar, q2-....
    print(models_checkpoint)
    device = "cuda" if  torch.cuda.is_available() else "cpu"

    #device = "cpu"
    images_path = args.image_path # path del test set 
    #savepath = args.result_path # path dove salvare i risultati 

    image_list = [os.path.join(images_path,f) for f in listdir(images_path)]
    


    dict_model_list =  {} #  inizializzo i modelli 

    # RICORDARSI DI METTERE I BASE QUANDO ARRIVERA' il MOMENTO!!!!!!
    for i, check in enumerate(models_checkpoint):  # per ogni cjeckpoint, salvo il modello nostro con la chiave q1-bmshj2018-sos ed il modello base con la chiave q1-bmshj2018-base (il modello base non ha checkpoint perchè lo prendo online)
        if True: #"00320" in check or "09500" in check or "04000" in check:
            name_sos = check.split("-")[0] + "-" + check.split("-")[1] # 00670 
            print("name_sos è il segddduente: ",name_sos)
            dict_model_list[name_sos] = check
        
    print("faccio il loading dei modelli")
    res = load_models(dict_model_list,  models_path, device, image_models) # carico i modelli e faccio update 
    

    metrics = eval_models(res,image_list , device,args.entropy_estimation) #faccio valutazione dei modelli





    
if __name__ == "__main__":

    wandb.init(project="prova", entity="albertopresta")   
    main(sys.argv[1:])
