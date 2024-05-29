
import sys
import wandb
import torch
import torch.optim as optim
from compress.training import train_one_epoch_gate, test_epoch_gate, DistorsionLoss,GateLoss, MssimLoss,AdaptersLoss,GateDistorsionLoss, DistorsionLoss, evaluate_base_model_gate
from compress.datasets import   AdapterDataset
from compress.utils.help_function import  configure_optimizers,  create_savepath, save_checkpoint_our, sec_to_hours, set_seed
from compress.utils.parser import parse_args_gate
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from compress.zoo import  get_gate_model
from compress.training.step import evaluate_base_model


def handle_trainable_pars(net, args, epoch):
    net.freeze_net()
    if args.train_baseline is False:

        if args.ms_ssim is True or args.oracle is True:
            net.handle_adapters_parameters()
        else:
            if args.training_policy == "gate":
                net.handle_gate_parameters()
            elif args.training_policy in ("e2e","e2e_mse"): #and args.starting_epoch < epoch:
                net.handle_adapters_parameters()
                if epoch < args.starting_epoch:
                    net.handle_gate_parameters()
            elif args.training_policy == "adapter":
                net.handle_adapters_parameters()
            elif args.training_policy == "fgta":
                if args.starting_epoch > epoch:
                    print("solo il gate")
                    net.handle_gate_parameters(re_grad = True)
                    net.handle_adapters_parameters(re_grad = False)
                else:
                    net.handle_gate_parameters(re_grad = False)
                    net.handle_adapters_parameters(re_grad = True)
            else:
                raise ValueError("unrecognized policy")
    else: 
        net.pars_decoder(starting = 5)  


        


def main(argv):
    args = parse_args_gate(argv)
    print(args,"cc")

    
    set_seed(seed = args.seed)
    device = "cuda" if  torch.cuda.is_available() else "cpu"
    num_adapter = len(args.considered_classes)


    ##########################################   INITIALIZE DATASET ####################################
    ####################################################################################################
    ####################################################################################################
    
    train_transforms = transforms.Compose([ transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    valid_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size),  transforms.ToTensor()]) #transforms.CenterCrop(args.patch_size),
    test_transforms = transforms.Compose([transforms.ToTensor()])


    train_dataset = AdapterDataset(root = args.root + "/train", path = args.train_datasets, classes = args.considered_classes, transform= train_transforms, num_element = 4000)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=4,shuffle=True, pin_memory=(device == device),)
        

    valid_dataset = AdapterDataset(root = args.root + "/valid", path = args.valid_datasets,classes = args.considered_classes, transform= valid_transforms,num_element = 816)
    valid_dataloader = DataLoader(valid_dataset,batch_size= args.batch_size ,num_workers=4,shuffle=False, pin_memory=(device == device),)
        

    test_total_dataset = AdapterDataset(root = args.root + "/test", path  =  args.test_datasets, classes = args.considered_classes ,transform = test_transforms,num_element = 30,train = False)
        
    test_total_dataloader = DataLoader(test_total_dataset,batch_size= 1,num_workers=4,shuffle=False, pin_memory=(device == device),)
        
    net, modello_base  = get_gate_model(args, num_adapter,device) #se voglio allenare la baseline, questo è la baseline #dd

    res_base = evaluate_base_model(modello_base,args,device)

    """
    print("dataset checking")
    train_tens = torch.zeros(len(train_dataset.samples))
    for i in range(len(train_dataset.samples)):
        train_tens[i] = int(train_dataset.samples[i][1])
    print("----------------------------> train distribution ",torch.unique(train_tens, return_counts = True))
    train_tens = torch.zeros(len(valid_dataset.samples))
    for i in range(len(valid_dataset.samples)):
        train_tens[i] = int(valid_dataset.samples[i][1])
    print("----------------------------> valid distribution ",torch.unique(train_tens, return_counts = True))
    train_tens = torch.zeros(len(test_total_dataset.samples))
    for i in range(len(test_total_dataset.samples)):
        train_tens[i] = int(test_total_dataset.samples[i][1])
    print("--------------------->test distribution ",torch.unique(train_tens, return_counts = True))
    """


    if args.train_baseline:
        criterion = DistorsionLoss(lmbda = args.lmbda)
    elif args.ms_ssim:
        criterion = MssimLoss()
    else:
        if args.training_policy == "gate":
            criterion =  GateLoss()
        elif args.training_policy in ("e2e","e2e_mse"):
            criterion = GateDistorsionLoss(lmbda = args.lmbda, policy = args.training_policy)
        elif args.training_policy == "adapter":
            criterion = AdaptersLoss()
        elif args.training_policy == "fgta":
            criterion = GateDistorsionLoss(lmbda = args.lmbda)
        else:
            criterion = DistorsionLoss()


    optimizer, _ = configure_optimizers(net, args)
    lr_scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.patience)

    if args.restart_training:
        epoch = torch.load(args.pret_checkpoint)["epoch"]
        optimizer.load_state_dict(torch.load(args.pret_checkpoint , map_location=device)["optimizer"])
        optimizer.load_state_dict(torch.load(args.pret_checkpoint , map_location=device)["lr_scheduler"])
    else:
        epoch = 0

    counter = 0
    best_loss = float("inf")

    writing = args.writing_path 


    handle_trainable_pars(net,args, epoch)

    print("****************************************************************************************************************")
    print("****************************************************************************************************************")

    epoch_enc = 0

    for epoch in range(0, args.epochs):
        print("************************************************************************************************")
        print("*************************************** EPOCH  ",epoch," *****************************************")
        handle_trainable_pars(net, args, epoch)
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
        for nn,tt in net.named_parameters():
            if "original" in nn:
                print(tt.requires_grad)
        




        start = time.time()
        
        
        if model_tr_parameters > 0:
            counter = train_one_epoch_gate(net, criterion, train_dataloader, optimizer,args.training_policy,epoch,counter, oracle= args.oracle,train_baseline = args.train_baseline)

        loss_valid = test_epoch_gate(epoch, valid_dataloader, net, criterion,  args.training_policy,valid = True, oracle= args.oracle,train_baseline = args.train_baseline)
        lr_scheduler.step(loss_valid)


        # faccio vari test!

        _ = test_epoch_gate(epoch, test_total_dataloader, net, criterion, args.training_policy,  valid = False, oracle= args.oracle,train_baseline = args.train_baseline)
        

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
        

        if epoch%10==0 or (is_best and epoch > 200):

            net.update()
            res_ads = evaluate_base_model_gate(net, args,device, epoch_enc,num_adapter,oracle = args.oracle, train_baseline = args.train_baseline, writing  = writing) #writing
            epoch_enc += 1
            
        

        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH! ", epoch)

    
    
if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch -DCC-q1 #ssssssssssssss
    wandb.init(project="cheng_eval", entity="albertopresta")   
    main(sys.argv[1:])

