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


import sys
import wandb
import torch
import torch.optim as optim
from compress.training import train_one_epoch, test_epoch,  compress_with_ac, RateDistortionLoss ,  DistorsionLoss
from compress.datasets import   handle_dataset
from compress.utils.help_function import CustomDataParallel, configure_optimizers,  create_savepath, save_checkpoint_our, sec_to_hours, set_seed
from compress.utils.parser import parse_args
from compress.models.utils import get_model







def handle_trainable_pars(net, args):
    if args.training_policy in ("mse","rate"): 
        print("entro qua per sbloccare gli adapter")
        net.freeze_net()
        net.pars_adapter(re_grad = True)
        #net.pars_decoder(re_grad = args.unfreeze_decoder, st = args.level_dec_unfreeze)
        #net.parse_hyperprior(  unfreeze_hsa_loop= args.unfreeze_hsa_loop, unfreeze_hsa = args.unfreeze_hsa)
        # aggiungo l'adapter e lo sfreezo!
        #  

        if args.training_policy == "rate":
            net.pars_entropy_estimation()
    elif args.training_policy == "quantization":
        net.freeze_net()
        net.pars_adapter(re_grad = True) 











import time
def main(argv):
    args = parse_args(argv)
    print(args,"cc")

    
    set_seed(seed = args.seed)
    device = "cuda" if  torch.cuda.is_available() else "cpu"




    train_dataloader, valid_dataloader, test_dataloader, filelist = handle_dataset(args, device = device)





    """
    if args.model == "base":

        factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = True)
    else:
        factorized_configuration , gaussian_configuration = configure_latent_space_policy(args, device, baseline = False)
    print("gaussian configuration----- -fdddguuggffxssssxxx------>: ",gaussian_configuration)
    print("factorized configuration------>ceeccccssààcccc->: ",factorized_configuration)
    """





    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)

    net = get_model(args,device)
    net = net.to(device)





    print("*****************************************  PRIMA ADAPTER ***********************************************************************")
    net.update() #444
    #print("the filelist for compressinf is .",filelist)
    b = compress_with_ac(net, filelist, device, -1, loop=False)
    print("****************************************************************************************************************")
    print("****************************************************************************************************************")
    print("****************************************************************************************************************")



    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)



    if args.training_policy == "mse":
        print("entro qua che c'è mse distorsion loss")
        criterion =  DistorsionLoss()
        #if args.model != "decoder":
        #    net.modify_adapter(args, device) 
        net = net.to(device)        
    else:    
        criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0







    optimizer, aux_optimizer = configure_optimizers(net, args)
    print("hola!")
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=20)
    if args.scheduler == "plateau":
        lr_scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.patience)
    elif args.scheduler == "multistep":
        print("Multistep scheduler")
        lr_scheduler =optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,250,350,500,550], gamma=0.5)
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

    net.print_information()




    print("*****************************************  DOPO AGGIUNTA ADAPTER ***********************************************************************")
    net.update() #444
    #print("the filelist for compressinf is .",filelist)
    #b = compress_with_ac(net, filelist, device, -1, loop=False)
    print("****************************************************************************************************************")
    print("****************************************************************************************************************")


    for epoch in range(last_epoch, args.epochs):
        print("**************** epoch: ",epoch,". Counter: ",counter," ")
        previous_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}","    ",previous_lr)
        print("epoch ",epoch)

        print("##########################################################################################")
        print("vedo se i parametri che devono essere bloccati lo sono")
        for nn,tt in net.named_parameters():
            if "original" in nn:
                print(tt.requires_grad)







        print("************************************************************************************************")
        print("*************************************** EPOCH  ",epoch," *****************************************")
        model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        #if baseline is False:
        #    print(" freezed adapter parameterers: ",sum(p.numel() for p in net.adapter.parameters() if p.requires_grad == False))
        #    print(" trainable adapter parameterers: ",sum(p.numel() for p in net.adapter.parameters() if p.requires_grad))
        start = time.time()
        

        if model_tr_parameters > 0:
            counter = train_one_epoch(net, criterion, train_dataloader, optimizer,epoch, 1,counter)
 
        loss_valid = test_epoch(epoch, valid_dataloader, net, criterion,  valid = True)
        
        loss = test_epoch(epoch, test_dataloader, net, criterion,  valid = False)
        lr_scheduler.step(loss_valid)


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        filename, filename_best =  create_savepath(args, epoch)


        if epoch%10==0 or epoch == 399:
            net.update()
            compress_with_ac(net,filelist, device, epoch_enc)
            epoch_enc += 1
  





        if (is_best or epoch%5==0):
            net.update()
            save_checkpoint_our(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args":args
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
    #Enhanced-imagecompression-adapter-sketch
    wandb.init(project="Enhanced-imagecompression-adapter-sketch", entity="albertopresta")   
    main(sys.argv[1:])




