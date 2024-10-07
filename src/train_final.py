
import sys
import wandb
import torch
import torch.optim as optim
from compress.training import train_one_epoch_gate, test_epoch_gate,GateDistorsionLoss,  evaluate_base_model_gate
from compress.datasets import   AdapterDataset
from compress.utils.help_function import  configure_optimizers,  create_savepath, save_checkpoint_our, sec_to_hours, set_seed
from compress.utils.parser import parse_args_gate
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from compress.zoo import  get_gate_model
from compress.training.step import evaluate_base_model


def handle_trainable_pars(net, starting_epoch, epoch):
    net.freeze_net()
    net.handle_adapters_parameters()
    if epoch < starting_epoch:
        net.handle_gate_parameters()



        


def main(argv):
    args = parse_args_gate(argv)

    print(args,"cc")

    
    set_seed(seed = args.seed)
    device = "cuda" if  torch.cuda.is_available() else "cpu"
    num_adapter = len(args.considered_classes)

    project_name = "prova"


    if args.wandb_log:
        wandb.init(
            project='LIC-DA',
            name=f'{project_name}_seed-{args.seed}',
            config=args,
            entity="alberto-presta"
        )  


    ##########################################   INITIALIZE DATASET ####################################
    ####################################################################################################
    ####################################################################################################
    
    train_transforms = transforms.Compose([ transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    valid_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size),  transforms.ToTensor()]) #transforms.CenterCrop(args.patch_size),
    test_transforms = transforms.Compose([transforms.ToTensor()])


    train_dataset = AdapterDataset(base_root = args.root, # + "/train", 
                                   type = "train",
                                   classes = args.considered_classes, 
                                   transform= train_transforms, 
                                   num_element = 4000)
    
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=4,shuffle=True, pin_memory=(device == device),)
        

    valid_dataset = AdapterDataset(base_root = args.root, # + "/valid", 
                                   type = "valid",
                                   classes = args.considered_classes,
                                     transform= valid_transforms,
                                     num_element = 816)
    valid_dataloader = DataLoader(valid_dataset,batch_size= args.batch_size ,num_workers=4,shuffle=False, pin_memory=(device == device),)
        

    test_total_dataset = AdapterDataset(base_root = args.root,
                                        type = "test",
                                         classes = args.test_classes ,
                                         transform = test_transforms,
                                         num_element = 30)
        
    test_total_dataloader = DataLoader(test_total_dataset,batch_size= 1,num_workers=4,shuffle=False, pin_memory=(device == device),)
      
    net, modello_base  = get_gate_model(args, num_adapter,device) #se voglio allenare la baseline, questo è la baseline #dd

    res_base = evaluate_base_model(modello_base,args,device,considered_classes =args.test_classes)



    print("done")
    
    criterion = GateDistorsionLoss(lmbda = args.lmbda)
    optimizer, _ = configure_optimizers(net, args)
    lr_scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.patience)

    epoch = 0
    counter = 0
    best_loss = float("inf")

    args.writing = None if args.writing == "" else args.writing


    handle_trainable_pars(net,args.starting_epoch, epoch)
    model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
    print(" trainable parameters: ",model_tr_parameters)
    print(" freeze parameterss: ", model_fr_parameters)



    print("****************************************************************************************************************")
    print("****************************************************************************************************************")

    epoch_enc = 0

    for epoch in range(0, args.epochs):
        print("************************************************************************************************")
        print("*************************************** EPOCH  ",epoch," *****************************************")
        handle_trainable_pars(net, args.starting_epoch, epoch)
        model_tr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)

        print("check trainable parameters and frozen parameters")
        model_fr_parameters = sum(p.numel() for n,p in net.named_parameters() if "g_s" in n  and p.requires_grad is False)
        print("decoder: ",model_fr_parameters)
        model_fr_parameters = sum(p.numel() for n,p in net.named_parameters() if "gate" in n  and  p.requires_grad) #adapter
        print("gate: ",model_fr_parameters)   
        model_fr_parameters = sum(p.numel() for n,p in net.named_parameters() if "adapter" in n  and  p.requires_grad) #adapter
        print("adapter: ",model_fr_parameters)   
        print("**********************************  INFORMATION *****************************************************")
        net.print_information()    
        #for nn,tt in net.named_parameters():
        #    if "original" in nn:
        #        print(tt.requires_grad)
        




        start = time.time()
        
        
        if model_tr_parameters > 0:
            counter = train_one_epoch_gate(net, criterion, train_dataloader, optimizer,args.training_policy,epoch,counter, oracle= args.oracle,wandb_log=  args.wandb_log)

        loss_valid = test_epoch_gate(epoch, valid_dataloader, net, criterion,  args.training_policy,valid = True, oracle= args.oracle,wandb_log=  args.wandb_log)
        lr_scheduler.step(loss_valid)


        # faccio vari test!

        _ = test_epoch_gate(epoch, test_total_dataloader, net, criterion, args.training_policy,  valid = False, oracle= args.oracle,wandb_log=  args.wandb_log)
        

        is_best = loss_valid < best_loss
        best_loss = min(loss_valid, best_loss)
        filename, filename_best =  create_savepath(args, epoch, epoch_enc)



        if (is_best or epoch%5==0):
            net.update()
            save_checkpoint_our(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
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
        # learning rate stuf
        print("start the part related to beta")
        

        

        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH! ", epoch)

    
    
if __name__ == "__main__":

    main(sys.argv[1:])

