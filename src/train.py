# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import sys
import wandb
import torch
import torch.optim as optim
from compress.training import train_one_epoch, test_epoch,  compress_with_ac, RateDistortionLoss , AdapterLoss, DistorsionLoss, KnowledgeDistillationLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import os 

from compress.datasets import ImageFolder
from compress.zoo import models
from compress.utils.annealings import *
from compress.utils.help_function import CustomDataParallel, configure_optimizers, configure_latent_space_policy, create_savepath, save_checkpoint_our, sec_to_hours
from compress.utils.parser import parse_args
from torch.utils.data import Dataset
from compress.utils.plotting import plot_sos
from PIL import Image
from compress.models.utils import get_model






def handle_trainable_pars(net, args):
    if args.training_policy in ("adapter","mse","kd"): 
        net.freeze_net()
        net.pars_adapter(re_grad = True)
        net.pars_decoder(re_grad = args.unfreeze_decoder)
        # aggiungo l'adapter e lo sfreezo!
        #  

    elif args.training_policy == "quantization":
        print("io sono qui dentro? dovrei esserlo per forza")
        net.freeze_net()
        net.unfreeze_quantizer()
        net.pars_adapter(re_grad =False) 
        net.parse_hyperprior( re_grad_ha = args.unfreeze_ha, re_grad_hma = args.unfreeze_hma, re_grad_hsa = args.unfreeze_hsa)


    elif args.training_policy == "quantization_lrp":
        print("io sono qui dentro? dovrei esserlo per forza!")
        net.freeze_net()
        net.unfreeze_quantizer() 
        net.unfreeze_lrp()

    elif args.training_policy == "kd":
        print("ouuuu devo essere qua")
        net.freeze_net()
        #net.unfreeze_quantizer() 
        net.pars_adapter(re_grad = True) 
    elif args.training_policy == "residual":
        net.freeze_net() 
        net.unfreeze_residualModel()  
    elif args.training_policy == "only_lrp":         
        net.freeze_net()
        net.unfreeze_lrp()

def freeze_net(net):
    for n,p in net.named_parameters():
        p.requires_grad = False
        
    for p in net.parameters(): 
        p.requires_grad = False

def from_state_dict(cls, state_dict):

    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net

class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose(
        [ transforms.ToTensor()]
    )
        return transform(image)

    def __len__(self):
        return len(self.image_path)




def rename_key(key):
    """Rename state_dict key.rrrrrhhhr"""

    # Deal with modules trained wffffith DavvvtaParallel
    if key.startswith("module."):
        key = key[7:]
    if key.startswith('h_s.'):
        return None

    # ResidualBlockWithStride: 'downsample' -> 'skip'dd
    # if ".downsample." in key:
    #     return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters  ppppccc
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key

def load_pretrained(state_dict):
    """Convert sccctaddte_dicteee keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict


def modify_dictionary(check):
    res = {}
    ks = list(check.keys())
    for key in ks: 
        res[key[7:]] = check[key]
    return res


import time
def main(argv):
    args = parse_args(argv)
    print(args,"cc")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    

    device = "cuda" if  torch.cuda.is_available() else "cpu"



    

    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms, num_images=args.num_images_train)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True, pin_memory=(device == "cuda"),)

    valid_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    valid_dataset = ImageFolder(args.dataset, split="test", transform=valid_transforms, num_images=args.num_images_val)
    valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)




    #test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)sss
    test_dataset = TestKodakDataset(data_dir="/scratch/dataset/kodak")
    test_dataloader = DataLoader(test_dataset, batch_size=1,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    

    



    




    if args.model == "base":

        factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = True)
    else:
        factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = False)
    print("gaussian configuration----- -fdddguuggffxssssxxx------>: ",gaussian_configuration)
    print("factorized configuration------>ceeccccssààcccc->: ",factorized_configuration)






    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)

    net, baseline = get_model(args,device, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration )


    if args.training_policy == "kd":



        teacher_net = models["base"](N = args.dims_n, M = args.dims_m)
        checkpoint = torch.load(args.pret_checkpoint_teacher, map_location=device)
        if "entropy_bottleneck._quantized_cdf" in list(checkpoint.keys()):
            del checkpoint["entropy_bottleneck._offset"]
            del checkpoint["entropy_bottleneck._quantized_cdf"]
            del checkpoint["entropy_bottleneck._cdf_length"]
        if "gaussian_conditional._quantized_cdf" in list(checkpoint.keys()):
            del checkpoint["gaussian_conditional._offset"]
            del checkpoint["gaussian_conditional._quantized_cdf"]
            del checkpoint["gaussian_conditional._cdf_length"]
            del checkpoint["gaussian_conditional.scale_table"]
            
            

                
        teacher_net.load_state_dict(checkpoint)
        teacher_net.to(device) 
        teacher_net.update()
        


    net = net.to(device)



    sos = not baseline

    print("sos is the follwoing ",sos)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)




    if args.training_policy == "quantization":

        criterion = RateDistortionLoss(lmbda=args.lmbda)
        not_adapters = True
    elif args.training_policy == "kd":
        criterion = KnowledgeDistillationLoss(teacher_net,lmbda = args.lmbda)
        not_adapters = False

        if args.dim_adapter != 0 and args.model == "latent":
            net.modify_adapter(args, device) 
            net = net.to(device)
    elif args.training_policy == "mse":

        criterion =  DistorsionLoss()
        not_adapters = False


        # devo modificare il modello, al momento l'adapter non ha parametri allenabili 
        if args.dim_adapter != 0 and args.model == "latent":
            net.modify_adapter(args, device) 
            net = net.to(device)



    elif args.training_policy == "adapter" or args.training_policy == "only_lrp": # in questo caso alleno solo l'adapter 

        criterion = AdapterLoss()
        print("se sono entrato qua va benissimmo!")
        not_adapters = False
        
        # devo modificare il modello, al momento l'adapter non ha parametri allenabili 
        if args.dim_adapter != 0 and args.model == "latent":
            net.modify_adapter(args, device) 
            net = net.to(device)
            
    else:
       
        quantization_policy = None
        criterion = RateDistortionLoss(lmbda=args.lmbda)
        not_adapters = True
    last_epoch = 0



    optimizer, aux_optimizer = configure_optimizers(net, args, baseline)
    print("hola!")
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=20)
    if args.scheduler == "plateau":
        lr_scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=args.patience)
    elif args.scheduler == "multistep":
        print("Multistep scheduler")
        lr_scheduler =optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,250,350,500,550], gamma=0.3)
    elif args.scheduler == "steplr":
        print("multistep every stepsize")
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    counter = 0
    best_loss = float("inf")
    epoch_enc = 0
    previous_lr = optimizer.param_groups[0]['lr']







    model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
        
    print(" Ttrainable parameters prima : ",model_tr_parameters)
    print(" freeze parameters prima: ", model_fr_parameters)

    #net.unfreeze_quantizer
    #net.print_pars()

    epoch = 0

    handle_trainable_pars(net,args)

    model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
    print(" trainable parameters: ",model_tr_parameters)
    print(" freeze parameters: ", model_fr_parameters)

    filelist = [os.path.join("/scratch/dataset/kodak",f) for f in os.listdir("/scratch/dataset/kodak")]

    for epoch in range(last_epoch, args.epochs):
        print("**************** epoch: ",epoch,". Counter: ",counter," ",not_adapters)
        previous_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}","    ",previous_lr)
        print("epoch ",epoch)




        print("*********************************************************************************************")
        print("*************************************** EPOCH  ",epoch," *****************************************")
        model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        if baseline is False:
            print(" freezed adapter parameterers: ",sum(p.numel() for p in net.adapter.parameters() if p.requires_grad == False))
            print(" trainable adapter parameterers: ",sum(p.numel() for p in net.adapter.parameters() if p.requires_grad))
        start = time.time()
        

        if model_tr_parameters > 0:
            counter = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer,epoch, 1,counter,sos, not_adapters)
 
        loss_valid = test_epoch(epoch, valid_dataloader, net, criterion, sos, valid = True,not_adapters=not_adapters)
        
        loss = test_epoch(epoch, test_dataloader, net, criterion, sos, valid = False, not_adapters=not_adapters)
        lr_scheduler.step(loss_valid)


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        filename, filename_best =  create_savepath(args, epoch)


        if epoch > 10 and (epoch%50==0 or epoch == 599):
            net.update()
            compress_with_ac(net,filelist, device, epoch_enc)
            epoch_enc += 1
            if baseline is False and sos: 
                plot_sos(net, device)



        if is_best:
            net.update()



        if baseline: #and (is_best or epoch%25==0):
            if  (is_best or epoch%5==0):
                save_checkpoint_our(
                    {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename,
                filename_best
                )
        else:

            filename, filename_best =  create_savepath(args, epoch)

            if (is_best) or epoch%25==0:
                if sos:
                    save_checkpoint_our(
                            {
                                "epoch": epoch,
                                "state_dict": net.state_dict(),
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "factorized_configuration": net.factorized_configuration,
                                "gaussian_configuration":net.gaussian_configuration,
                                #"entropy_bottleneck_w":net.entropy_bottleneck.sos.w,
                                "gaussian_conditiona._w":net.gaussian_conditional.sos.w,

                        },
                        is_best,
                        filename,
                        filename_best    
                    )   
                else:
                    save_checkpoint_our(
                            {
                                "epoch": epoch,
                                "state_dict": net.state_dict(),
                                "loss": loss,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),

                        },
                        is_best,
                        filename,
                        filename_best    
                    )   



        print("log also the current leraning rate")

        log_dict = {
        "train":epoch,
        "train/leaning_rate": optimizer.param_groups[0]['lr']
        #"train/beta": annealing_strategy_gaussian.bet
        }

        wandb.log(log_dict)
        # learning rate stuff 
        print("start the part related to beta")
        
        

        
        

        

        #if epoch == 200:
            #net.freeze_quantizer()
        
        

        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)
  



      
        

if __name__ == "__main__":
    #KnowledgeDistillationImageCompression
    wandb.init(project="Knowldge-distillation", entity="albertopresta")   
    main(sys.argv[1:])




