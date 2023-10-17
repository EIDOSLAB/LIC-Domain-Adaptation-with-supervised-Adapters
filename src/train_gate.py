
import sys
import wandb
import torch
import torch.optim as optim
from compress.training import train_one_epoch_gate, test_epoch_gate, DistorsionLoss,GateLoss, MssimLoss, compress_with_ac,AdaptersLoss,GateDistorsionLoss, DistorsionLoss, evaluate_base_model_gate
from compress.datasets import   AdapterDataset, ImageFolder,  TestKodakDataset
from compress.utils.help_function import  configure_optimizers,  create_savepath, save_checkpoint_our, sec_to_hours, set_seed
from compress.utils.parser import parse_args_gate
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from compress.zoo import models
import os   

from compressai.zoo import    image_models

#os.environ['WANDB_CONFIG_DIR'] = "/src/" 


def evaluate_base_model(model, args,device):
    """
    Valuto la bontà del modello base, di modo da poter vedere se miglioraiamo qualcosadddd
    """

    model.to(device)
    model.update()
    test_transforms = transforms.Compose([transforms.ToTensor()])


    print("****************************************** SKETCH ****************************************************************")
    if "sketch" not in args.considered_classes:
        cons_classes = args.considered_classes + ["sketch"]
    else:
        cons_classes = args.considered_classes 
    painting = AdapterDataset(root = args.root + "/test", 
                              path  =  ["_sketch_.txt"],
                              classes = cons_classes, 
                              transform = test_transforms,
                               num_element = 25,
                              train = False)
    painting_f = painting.samples
    painting_filelist = []
    for i in range(len(painting_f)):
        painting_filelist.append(painting_f[i][0])
    psnr, bpp = compress_with_ac(model, painting_filelist, device, -1, loop=False, writing= args.writing + "sketch_")
    print(psnr,"  ",bpp)


    print("******************************************  CLIPART ****************************************************************")
    if "clipart" not in args.considered_classes:
        cons_classes = args.considered_classes + ["clipart"]
    else:
        cons_classes = args.considered_classes 
    painting = AdapterDataset(root = args.root + "/test", 
                              path  =  ["_clipart_.txt"],
                              classes = cons_classes,
                                transform = test_transforms,
                                num_element=25,
                                train = False)
    painting_f = painting.samples
    painting_filelist = []
    for i in range(len(painting_f)):
        painting_filelist.append(painting_f[i][0])
    psnr, bpp = compress_with_ac(model, painting_filelist, device, -1, loop=False, writing= args.writing + "comic_")
    print(psnr,"  ",bpp)
    print("******************************************  KODAK ****************************************************************")
    painting = AdapterDataset(root = args.root + "/test", 
                              path  =  ["_kodak_.txt"],
                              classes = args.considered_classes,
                                transform = test_transforms,
                                num_element=25,
                                train = False)
    painting_f = painting.samples
    painting_filelist = []
    for i in range(len(painting_f)):
        painting_filelist.append(painting_f[i][0])
    psnr, bpp = compress_with_ac(model, painting_filelist, device, -1, loop=False, writing= args.writing + "kodak_")
    print(psnr,"  ",bpp)

    print("******************************************  INFO ****************************************************************")
    if "comic" not in args.considered_classes:
        cons_classes = args.considered_classes + ["infograph"]
    else:
        cons_classes = args.considered_classes 
    painting = AdapterDataset(root = args.root + "/test", 
                              path  =  ["_infograph_.txt"],
                              classes = args.considered_classes,
                                transform = test_transforms,
                                num_element=25,
                                train = False)
    painting_f = painting.samples
    painting_filelist = []
    for i in range(len(painting_f)):
        painting_filelist.append(painting_f[i][0])
    psnr, bpp = compress_with_ac(model, painting_filelist, device, -1, loop=False, writing= args.writing + "infographics_")
    print(psnr,"  ",bpp)
    print("******************************************  PAINTING ****************************************************************")

    if "painting" not in args.considered_classes:
        cons_classes = args.considered_classes + ["painting"]
    else:
        cons_classes = args.considered_classes 
    painting = AdapterDataset(root = args.root + "/test", 
                              path  =  ["_painting_.txt"],
                              classes = cons_classes,
                                transform = test_transforms,
                                num_element=25,
                                train = False)
    painting_f = painting.samples
    painting_filelist = []
    for i in range(len(painting_f)):
        painting_filelist.append(painting_f[i][0])
    psnr, bpp = compress_with_ac(model, painting_filelist, device, -1, loop=False, writing= args.writing + "watercolor_")
    print(psnr,"  ",bpp)

    print("******************************************  PAINTING ****************************************************************")

    if "quickdraw" not in args.considered_classes:
        cons_classes = args.considered_classes + ["quickdraw"]
    else:
        cons_classes = args.considered_classes 
    painting = AdapterDataset(root = args.root + "/test", 
                              path  =  ["_quickdraw_.txt"],
                              classes = cons_classes,
                                transform = test_transforms,
                                num_element=25,
                                train = False)
    painting_f = painting.samples
    painting_filelist = []
    for i in range(len(painting_f)):
        painting_filelist.append(painting_f[i][0])
    psnr, bpp = compress_with_ac(model, painting_filelist, device, -1, loop=False, writing= args.writing + "watercolor_")
    print(psnr,"  ",bpp)

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


        

def rename_key_for_adapter(key, stringa, nuova_stringa):
    if key.startswith(stringa):
        key = nuova_stringa # nuova_stringa  + key[6:]
    return key


def from_state_dict(cls, state_dict):
    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net

def get_gate_model(args,num_adapter, device):

    if args.name_model == "cheng":
        
        
        modello_base =  image_models["cheng2020-attn"](quality=args.quality, metric="mse", pretrained=True, progress=False)
        modello_base.update()
        modello_base.to(device)   

        state_dict = modello_base.state_dict() 


        state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.9.original_model_weights.0.weight", stringa = "g_s.9.0.weight" ): v for k, v in state_dict.items()}
        state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.9.original_model_weights.0.bias", stringa = "g_s.9.0.bias" ): v for k, v in state_dict.items()}

        print("**************************************************** MODELLO BASE ******************************************************")
        print("************************************************************************************************************************")
        for k in list(state_dict.keys()):
            if "g_s" in k:
                print(k)


        net = models["cheng"](N=args.N,
                                dim_adapter_attn = args.dim_adapter_attn,
                                stride_attn = args.stride_attn,
                                kernel_size_attn = args.kernel_size_attn,
                                padding_attn = args.padding_attn,
                                type_adapter_attn = args.type_adapter_attn,
                                position_attn = args.position_attn,
                                num_adapter = num_adapter,
                                aggregation = args.aggregation,
                                std = args.std,
                                mean = args.mean,                              
                                bias = True,

        )

        print("************************************************************************   MODELLO NUOVO *****************************************************")
        print("***********************************************************************************************************************************************")
        for k in list(net.state_dict().keys()):
            if "g_s" in k:
                print(k)

        
        return net, modello_base




    elif args.name_model == "WACNN":
        if args.train_baseline:
            checkpoint = torch.load(args.pret_checkpoint , map_location=device)
            net = from_state_dict(models["base"], checkpoint)
            net.to(device)
            net.update()
            return net, net
        else:
            if args.origin_model == "base":
                checkpoint = torch.load(args.pret_checkpoint , map_location=device)#["state_dict"]
            else:
                checkpoint = torch.load(args.pret_checkpoint , map_location=device)["state_dict"]
            modello_base = from_state_dict(models[args.origin_model], checkpoint)
            modello_base.update()
            modello_base.to(device) 







            net = models["gate"](N = args.N,
                                M =args.M,
                                dim_adapter_attn = args.dim_adapter_attn,
                                stride_attn = args.stride_attn,
                                kernel_size_attn = args.kernel_size_attn,
                                padding_attn = args.padding_attn,
                                type_adapter_attn = args.type_adapter_attn,
                                position_attn = args.position_attn,
                                num_adapter = num_adapter,
                                aggregation = args.aggregation,
                                std = args.std,
                                mean = args.mean,                              
                                bias = True,
                                skipped = args.skipped
                                    ) 


            state_dict = modello_base.state_dict()
            if args.origin_model == "base":
                state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.6.original_model_weights.weight", stringa = "g_s.6.weight" ): v for k, v in state_dict.items()}
                state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.6.original_model_weights.bias", stringa = "g_s.6.bias" ): v for k, v in state_dict.items()}
                state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.8.original_model_weights.weight", stringa = "g_s.8.weight" ): v for k, v in state_dict.items()}
                state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.8.original_model_weights.bias", stringa = "g_s.8.bias" ): v for k, v in state_dict.items()}
                

            if args.pret_checkpoint_gate != "none":
                print("ENTRO QUA SE VOGLIO IL GATE PRE ALLENATO!!!!!!!!")
                gate_dict = torch.load(args.pret_checkpoint_gate , map_location=device)["state_dict"]
                for k in list(gate_dict.keys()):
                    if "gate." in k:
                        state_dict[k] = gate_dict[k]

            

            
            info = net.load_state_dict(state_dict, strict=False)
            net.to(device)
            return net, modello_base



def main(argv):
    args = parse_args_gate(argv)
    print(args,"cc")

    
    set_seed(seed = args.seed)
    device = "cuda" if  torch.cuda.is_available() else "cpu"


    # numero di classsi

    num_adapter = len(args.considered_classes)
    # gestisco i dataset 


    
    train_transforms = transforms.Compose([ transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    valid_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size),  transforms.ToTensor()]) #transforms.CenterCrop(args.patch_size),
    test_transforms = transforms.Compose([transforms.ToTensor()])

    # args.root + "/train"
    train_dataset = AdapterDataset(root = args.root + "/train", path = args.train_datasets, classes = args.considered_classes, transform= train_transforms, num_element = 25000)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=4,shuffle=True, pin_memory=(device == device),)
        

    valid_dataset = AdapterDataset(root = args.root + "/valid", path = args.valid_datasets,classes = args.considered_classes, transform= valid_transforms,num_element = 816)
    valid_dataloader = DataLoader(valid_dataset,batch_size= args.batch_size ,num_workers=4,shuffle=False, pin_memory=(device == device),)
        

    test_total_dataset = AdapterDataset(root = args.root + "/test", path  =  args.test_datasets, classes = args.considered_classes ,transform = test_transforms,num_element = 25,train = False)
        
    test_total_dataloader = DataLoader(test_total_dataset,batch_size= 1,num_workers=4,shuffle=False, pin_memory=(device == device),)
        
    net, modello_base  = get_gate_model(args, num_adapter,device) #se voglio allenare la baseline, questo è la baseline

    evaluate_base_model(modello_base,args,device)

    
    
    print("CONTROLLO DATASET!!!!!")
    
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

        optimizer.load_state_dict(torch.load(args.pret_checkpoint , map_location=device)["optimizer"])
        optimizer.load_state_dict(torch.load(args.pret_checkpoint , map_location=device)["lr_scheduler"])


    counter = 0
    best_loss = float("inf")

    writing = "/scratch/KD/devil2022/results/adapters/q5/5_classes/" + "_" + str(args.lmbda)

    if args.restart_training is False:
        epoch = 0
    else:
        epoch = torch.load(args.pret_checkpoint)["epoch"]

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
        print("vedo se i parametri che devono essere bloccati lo sono")
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


        if epoch%5==0 or (is_best and epoch > 50):

            net.update()
            evaluate_base_model_gate(net, args,device, epoch_enc,num_adapter,oracle = args.oracle, train_baseline = args.train_baseline, writing  = None)
            epoch_enc += 1
            
        
        

        
        

        

        #if epoch == 200:
            #net.freeze_quantizer()
        
        

        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH! ", epoch)
    
    
if __name__ == "__main__":
    #Enhanced-imagecompression-adapter-sketch -DCC-q1 #ssssssssssss
    wandb.init(project="DCC-q5-DomainNet", entity="albertopresta")   
    main(sys.argv[1:])

