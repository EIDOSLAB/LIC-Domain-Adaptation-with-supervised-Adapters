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

from compressai.zoo import    image_models
def rename_key_for_adapter(key, stringa, nuova_stringa):
    if key.startswith(stringa):
        key = nuova_stringa # nuova_stringa  + key[6:]
    return key


def from_state_dict(cls, state_dict):
    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net



def read_image(filepath):
    #assert filepath.is_file()
    img = Image.open(filepath)
    img = img.convert("RGB")
    return img


def get_model_for_evaluation(model, pret_checkpoint, device, quality = None):



    if "devil2022" in pret_checkpoint: #args.name_model == "WACNN":
        checkpoint = torch.load(pret_checkpoint , map_location=device)
        if model == "base":
            checkpoint = torch.load(pret_checkpoint , map_location=device)#["state_dict"]

            net = from_state_dict(models[model], checkpoint)
            net.update()
            net.to(device) 
            return net
        else:

            state_dict = checkpoint["state_dict"]
            args = checkpoint["args"]








            net = models[model](N = args.N,
                                M =args.M,
                                dim_adapter_attn = args.dim_adapter_attn,
                                stride_attn = args.stride_attn,
                                kernel_size_attn = args.kernel_size_attn,
                                padding_attn = args.padding_attn,
                                type_adapter_attn = args.type_adapter_attn,
                                position_attn = args.position_attn,
                                num_adapter = 3 ,#len(args.considered_classes),
                                aggregation = args.aggregation,
                                std = args.std,
                                mean = args.mean,                              
                                bias = True, #args.bias,
                                skipped = False
                                    ) 
        print("args.aggregation---------> ",args.aggregation)
        info = net.load_state_dict(state_dict, strict=True)
        net.to(device)
        return net
    else:
        if model == "base":
            qual = int(quality[1])
            print("la qualità è: ",qual)
            net =  image_models["cheng2020-attn"](quality=qual, metric="mse", pretrained=True, progress=False)
            net.update()
            net.to(device)   
            return net
        else:
            checkpoint = torch.load(pret_checkpoint , map_location=device)
            state_dict = checkpoint["state_dict"]
            args = checkpoint["args"]



            net = models["cheng"](N=args.N,
                                    dim_adapter_attn = args.dim_adapter_attn,
                                    stride_attn = args.stride_attn,
                                    kernel_size_attn = args.kernel_size_attn,
                                    padding_attn = args.padding_attn,
                                    type_adapter_attn = args.type_adapter_attn, #ssss
                                    position_attn = args.position_attn,
                                    num_adapter = 3,
                                    aggregation = args.aggregation,
                                    std = args.std,
                                    mean = args.mean,                              
                                    bias = True,

            )

            info = net.load_state_dict(state_dict, strict=True)
            net.update()
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

    img = img.convert("RGB")
    return transforms.ToTensor()(img) 







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




    task = args.task
    num_element = args.num_element 
    path_task = "_" + task + "_.txt"
    quality = args.quality
    classes = "3_classes"
    dat = "_"
    root = "/scratch/dataset/domain_adapter/MixedImageSets/test"
    painting_filelist = []
    if "bam" not in task:

        test_transforms = transforms.Compose([transforms.ToTensor()])
        painting = AdapterDataset(root = root, path  =  [path_task], classes = ["natural","sketch","comic","quickdraw","infograph","watercolor","clipart","infograph","iam_document", "documents"],transform = test_transforms,train = False, num_element= num_element) #ddddddd
        painting_f =painting.samples
        oracles = []
        for i in range(len(painting_f)):
            if "b04-075" not in painting_f[i][0] and "b04-074" not in  painting_f[i][0]:
                painting_filelist.append(painting_f[i][0])
                oracles.append(int(painting_f[i][1]))
    else:

        bam_path = "/scratch/dataset/bam_dataset/splitting/_" + task + "_.txt" #_bam_comic_.txt"
        bam = "/scratch/dataset/bam_dataset/bam/"
        file_d = open(bam_path,"r") 
        Lines = file_d.readlines()
        for i,lines in enumerate(Lines):
            painting_filelist.append(bam + lines[:-1])


        oracles = []
        for i in range(len(painting_filelist)):
            
            oracles.append(2)

        print("lunghezza del bam",len(painting_filelist))
 

    print("linghezza del sample: ",len(painting_filelist))
    print("**************************************")
    print(painting_filelist)


   


    # IAM DOCUMENT 
    #iam_path = "/scratch/dataset/domain_adapter/iam_document"
    #painting_filelist = [os.path.join(iam_path,f) for f in os.listdir(iam_path)]
    #oracles = []
    #for i in range(len(painting_filelist)):
    #    oracles.append(1)






    


   


    
    
    mode = "gate"
    save_images = None #"/scratch/KD/devil2022/results/reconstructions/" + task + "/png/" + mode + "/" + quality #ss
    writing_path = args.writing + "/results/writings/mixture" #"/scratch/KD/devil2022/
    writing = os.path.join(writing_path,task, mode, quality + "/")
    


    pret_checkpoint = args.pret_checkpoint #("/scratch/KD/devil2022/adapter/mixture"
    path_models = os.path.join(pret_checkpoint,classes,quality) #DomainNet #sssssss
    
    
    list_models = [os.path.join(path_models,f) for f in os.listdir(path_models) if "top1" not in f and "oracle" not in f]
    
    for m in list_models:

        net = get_model_for_evaluation(mode,m,device)
        if mode == "gate":
            psnr, bpp = compress_with_ac_gate(net, 
                                        painting_filelist, 
                                        device, 
                                        -1, 
                                        3,
                                        loop=False, 
                                        name = dat + "_" + quality + "_" + task +  "_", 
                                        oracles= None ,# [] ,#oracles, 
                                        train_baseline= False,
                                        writing =writing,
                                        save_images = save_images)
        else:
            psnr,bpp =  compress_with_ac(net, painting_filelist, device, -1, loop=False,name =  quality + "_" + task + "_" ,  save_images = save_images, writing = writing)
        print(mode,": ",psnr,"  ",bpp)

    
    
    path_models = os.path.join("/scratch/universal-dic/weights",quality)
    mode = "base"
    

    save_images = None #"/scratch/KD/devil2022/results/reconstructions/" + task + "/png/" +mode + "/" + quality
    writing_path = args.writing + "/results/writings/mixture"
    writing = os.path.join(writing_path,task, mode, quality + "/") #dddddddddddd

    list_models = [os.path.join(path_models,f) for f in os.listdir(path_models)]
    for m in list_models:

        net = get_model_for_evaluation(mode,m,device,  quality = args.quality)
        if mode == "gate": #ssss
            psnr, bpp = compress_with_ac_gate(net, 
                                        painting_filelist, 
                                        device, 
                                        -1, 
                                        3,
                                        loop=False, 
                                        name =  quality + "_" + task + "_",  #sssssss
                                        oracles= None, 
                                        train_baseline= False,
                                        writing =writing,
                                        save_images = save_images)
        else:
            psnr,bpp =  compress_with_ac(net, painting_filelist, device, -1, loop=False,name =  quality + "_" + task + "_",  save_images = save_images, writing = writing)
        print(mode,": ",psnr,"  ",bpp)  





    
if __name__ == "__main__":

    wandb.init(project="prova", entity="albertopresta")   
    main(sys.argv[1:])
