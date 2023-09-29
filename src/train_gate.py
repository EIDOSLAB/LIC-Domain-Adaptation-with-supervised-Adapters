
import sys
import wandb
import torch
import torch.optim as optim
from compress.training import train_one_epoch_gate, test_epoch_gate, GateLoss, compress_with_ac, GateDistorsionLoss, DistorsionLoss, compress_with_ac_gate
from compress.datasets import   AdapterDataset
from compress.utils.help_function import CustomDataParallel, configure_optimizers,  create_savepath, save_checkpoint_our, sec_to_hours, set_seed
from compress.utils.parser import parse_args_gate
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from compress.zoo import models



def evaluate_base_model(model, args,device):
    """
    Valuto la bontà del modello base, di modo da poter vedere se miglioraiamo qualcosa
    """

    model.to(device)
    model.update()

    test_transforms = transforms.Compose([transforms.ToTensor()])
    print("***************************************** KODAK *************************************************")
    kodak = AdapterDataset(root = args.root, path  =  ["test_kodak.txt"], transform = test_transforms)
    kodak_f = kodak.samples
    kodak_filelist = []
    for i in range(len(kodak_f)):
        kodak_filelist.append(kodak_f[i][0])
    psnr, bpp = compress_with_ac(model, kodak_filelist, device, -1, loop=False, writing= args.writing + "kodak_")
    print(psnr,"  ",bpp)

    print("****************************************** CLIC ****************************************************************")
    clic = AdapterDataset(root = args.root, path  =  ["test_clic.txt"], transform = test_transforms)
    clic_f = clic.samples
    clic_filelist = []
    for i in range(len(clic_f)):
        clic_filelist.append(clic_f[i][0])
    psnr, bpp = compress_with_ac(model, clic_filelist, device, -1, loop=False, writing= args.writing + "clic_")
    print(psnr,"  ",bpp)

    print("****************************************** sketch ****************************************************************")
    sketch = AdapterDataset(root = args.root, path  =  ["test_sketch.txt"], transform = test_transforms)
    sketch_f = sketch.samples
    sketch_filelist = []
    for i in range(len(sketch_f)):
        sketch_filelist.append(sketch_f[i][0])
    psnr, bpp = compress_with_ac(model, sketch_filelist, device, -1, loop=False, writing= args.writing + "sketch_")
    print(psnr,"  ",bpp)

    print("****************************************** clipart ****************************************************************")
    clipart = AdapterDataset(root = args.root, path  =  ["test_clipart.txt"], transform = test_transforms)
    clipart_f = clipart.samples
    clipart_filelist = []
    for i in range(len(clipart_f)):
        clipart_filelist.append(clipart_f[i][0])
    psnr, bpp = compress_with_ac(model, clipart_filelist, device, -1, loop=False, writing= args.writing + "clipart_")
    print(psnr,"  ",bpp)



def handle_trainable_pars(net, args, epoch):
    net.freeze_net()
    if args.training_policy == "gate":
        print("porca troia ")
        net.unfreeze_gate()
    elif args.training_policy == "e2e": #and args.starting_epoch < epoch:
        net.unfreeze_gate()
        if args.starting_epoch < epoch:
            net.unfreeze_adapters()
    elif args.training_policy == "only_adapters":
        net.unfreeze_adapters()
    else:
        raise ValueError("unrecognized policy")
    


        

def rename_key_for_adapter(key, stringa, nuova_stringa):
    if key.startswith(stringa):
        key = nuova_stringa # nuova_stringa  + key[6:]
    return key


def from_state_dict(cls, state_dict):
    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net

def get_gate_model(args,device):

    checkpoint = torch.load(args.pret_checkpoint , map_location=device)#["state_dict"]
    print("INIZIO STATE DICT")
    modello_base = from_state_dict(models["base"], checkpoint)
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
                        num_adapter = args.num_adapter,
                        aggregation = args.aggregation,
                        std = args.std,
                        mean = args.mean,                              
                        bias = args.bias,
                              ) 
    """
    print("*********************************************************************************")
    print("************************************ NUOVO MODELLO ******************************************22222***")
    print("*********************************************************************************")
    for k in list(net.state_dict().keys()):
        if "g_s" in k or "g_a" in k:
            print(k)
    """





    state_dict = modello_base.state_dict()
    state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.6.original_model_weights.weight", stringa = "g_s.6.weight" ): v for k, v in state_dict.items()}
    state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.6.original_model_weights.bias", stringa = "g_s.6.bias" ): v for k, v in state_dict.items()}
    state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.8.original_model_weights.weight", stringa = "g_s.8.weight" ): v for k, v in state_dict.items()}
    state_dict = {rename_key_for_adapter(key = k, nuova_stringa = "g_s.8.original_model_weights.bias", stringa = "g_s.8.bias" ): v for k, v in state_dict.items()}
    
    """
    print("*********************************************************************************")
    print("************************************ VECCHIO  MODELLO MODIFICATO *********************************************")
    print("*********************************************************************************")
    for k in list(state_dict.keys()):
        if "g_s" in k or "g_a" in k:
            print(k)
    """
    
    
    info = net.load_state_dict(state_dict, strict=False)
    net.to(device)
    return net, modello_base



def main(argv):
    args = parse_args_gate(argv)
    print(args,"cc")

    
    set_seed(seed = args.seed)
    device = "cuda" if  torch.cuda.is_available() else "cpu"

    # gestisco i dataset 

    train_transforms = transforms.Compose([ transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    valid_transforms = transforms.Compose([  transforms.ToTensor()]) #transforms.CenterCrop(args.patch_size),
    test_transforms = transforms.Compose([transforms.ToTensor()])


    train_dataset = AdapterDataset(root = args.root, path = args.train_datasets, transform= train_transforms)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=4,shuffle=True, pin_memory=(device == device),)
    

    valid_dataset = AdapterDataset(root = args.root, path = args.valid_datasets, transform= valid_transforms)
    valid_dataloader = DataLoader(valid_dataset,batch_size= 1 ,num_workers=4,shuffle=False, pin_memory=(device == device),)
    

    test_total_dataset = AdapterDataset(root = args.root, path  =  args.test_datasets, transform = test_transforms)
    
    test_total_dataloader = DataLoader(test_total_dataset,batch_size= 1,num_workers=4,shuffle=False, pin_memory=(device == device),)
    
    net, modello_base = get_gate_model(args, device)

    #evaluate_base_model(modello_base,args,device)


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


    print("---------------------------->test distribution ",torch.unique(train_tens, return_counts = True))





    if args.training_policy == "gate":
        criterion =  GateLoss()
    elif args.training_policy == "e2e":
        criterion = GateDistorsionLoss(lmbda = args.lmbda)
    else:
        criterion = DistorsionLoss()


    optimizer, _ = configure_optimizers(net, args)

    lr_scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=args.patience)


    counter = 0
    best_loss = float("inf")


    epoch = 0

    handle_trainable_pars(net,args, epoch)

    print("****************************************************************************************************************")
    print("****************************************************************************************************************")


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
            counter = train_one_epoch_gate(net, criterion, train_dataloader, optimizer,args.training_policy,epoch,counter)

        loss_valid = test_epoch_gate(epoch, valid_dataloader, net, criterion,  args.training_policy,valid = True)
        lr_scheduler.step(loss_valid)


        # faccio vari test!

        _ = test_epoch_gate(epoch, test_total_dataloader, net, criterion, args.training_policy,  valid = False)
        

        is_best = loss_valid < best_loss
        best_loss = min(loss_valid, best_loss)
        filename, filename_best =  create_savepath(args, epoch)



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
    wandb.init(project="Gate-training", entity="albertopresta")   
    main(sys.argv[1:])

