import torch 

import wandb
from compress.utils.help_function import compute_msssim, compute_psnr
import torch.nn.functional as F 
from PIL import Image
from torchvision import transforms
import math
from compress.datasets import AdapterDataset
from compressai.ops import compute_padding
from sklearn.metrics import f1_score

def read_image(filepath, adapt =False):
    #assert filepath.is_file()
    img = Image.open(filepath)
    
    if adapt:
        i =  img.size
        i = 256, 256#i[0]//2, i[1]//2
        img = img.resize(i)
    img = img.convert("RGB")
    return transforms.ToTensor()(img) 


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


def train_one_epoch_gate(model, criterion, train_dataloader, optimizer,training_policy, epoch, counter, oracle, train_baseline):


    model.train()
    device = next(model.parameters()).device



    
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    gate_loss = AverageMeter()


    predictions = []
    labels = []
    total = 0
    correct = 0


    for i, (d,cl)  in enumerate(train_dataloader):


        counter += 1
        d = d.to(device)
        cl = cl.to(device)
        optimizer.zero_grad()
        #if aux_optimizer is not None: #ddd
        #    aux_optimizer.zero_grad()

        orac = cl if oracle else None

        if training_policy != "gate":
            if train_baseline:
                out_net = model(d)#model(d, oracle = orac)
            else:
                out_net = model(d, oracle = orac)
        else:
             out_net = model.forward_gate(d)
        
        if train_baseline is False:
            out_criterion = criterion(out_net, (d,cl))
        else:
            out_criterion = criterion(out_net, d)

        out_criterion["loss"].backward()
        optimizer.step()

        loss.update(out_criterion["loss"].clone().detach())
        if training_policy != "gate":
            mse_loss.update(out_criterion["mse_loss"].clone().detach())
            bpp_loss.update(out_criterion["bpp_loss"].clone().detach())

        if train_baseline is False:
            gate_loss.update(out_criterion["CrossEntropy"].clone().detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        if train_baseline is False:
            _, predicted = torch.max(F.softmax(out_net["logits"]), 1) # predicted
            predictions.extend(predicted.cpu().numpy().tolist())
            labels.extend(cl.cpu().numpy().tolist())

            total += cl.size(0)
            correct += (predicted == cl).sum().item()

        #if aux_optimizer is not None:
        #    aux_loss = model.aux_loss()
        #    aux_loss.backward()
        #    aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'


            )

        if train_baseline is False:
            wandb.log({"train_batch":counter,"train_batch/crossentropy":out_criterion["CrossEntropy"].clone().detach().item(),})


        wandb_dict = {
            "train_batch":counter,
            "train_batch/losses_batch":out_criterion["loss"].clone().detach().item(),
        }

        wandb.log(wandb_dict)

        if training_policy != "gate":

            wand_dict = {
                "train_batch": counter,
                "train_batch/mse":out_criterion["mse_loss"].clone().detach().item(),

            }
            wandb.log(wand_dict)

    accuracy = 100 * correct / (total + 1e-6)

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,


    }

    if train_baseline is False:
        wandb.log({"train":epoch,"train/crossentropy":gate_loss.avg,"train/accuracy":accuracy })

    if training_policy != "gate":

        log_dict = {
            "train":epoch,
            "train/mse": mse_loss.avg,
            }
            
        wandb.log(log_dict)
    return counter





def test_epoch_gate(epoch, test_dataloader, model, criterion, training_policy, valid, oracle, train_baseline ):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    mse_loss = AverageMeter()


    psnr = AverageMeter()
    ssim = AverageMeter()
    gate_loss = AverageMeter()



    total = 0
    correct = 0

    predictions = []
    labels = []

    with torch.no_grad():
        for (d,cl) in test_dataloader:

            d = d.to(device)
            cl = cl.to(device)

            h, w = d.size(2), d.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6ddd strides of 2
            d_padded = F.pad(d, pad, mode="constant", value=0)

            orac = cl if oracle else None
            if training_policy != "gate":
                if train_baseline:
                    out_net = model(d_padded)# model(d_padded, oracle = orac)
                else:
                    out_net = model(d_padded, oracle = orac)
            else:
                out_net = model.forward_gate(d_padded)
            
            out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)

            if train_baseline is False:
                out_criterion = criterion(out_net, (d,cl))
            else:
                out_criterion = criterion(out_net, d)



            #if valid:
            if train_baseline is False:
                _, predicted = torch.max(F.softmax(out_net["logits"]), 1) # predicted
                predictions.extend(predicted.cpu().numpy().tolist())
                labels.extend(cl.cpu().numpy().tolist())
                total += cl.size(0)
                correct += (predicted == cl).sum().item()
            
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

            if training_policy != "gate":
                psnr.update(compute_psnr(d, out_net["x_hat"]))
                ssim.update(compute_msssim(d, out_net["x_hat"]))
            
            if train_baseline is False:
                gate_loss.update(out_criterion["CrossEntropy"])


    accuracy = 100 * correct / (total + 1e-6)
    if train_baseline is False:
        f1 = f1_score(labels, predictions, average = 'weighted')
    #cm = confusion_matrix(labels, predictions)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |",
        f"\taccuracy: {accuracy:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"

    )





    if valid is False:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"

        )


        if train_baseline is False:
            log_dict =  {
                "test":epoch,
                "test/accuracy":accuracy,
                "test/crossentropy":gate_loss.avg,
                "test/f1_score":f1
            }
            wandb.log(log_dict)

            wandb.log({
                "plot_test":epoch,
                "plot_test/conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=labels, preds=predictions,
                            class_names=["natural","sketch","clipart","painting"])})




        log_dict = {
        "test":epoch,
        "test/loss": loss.avg,
        "test/mse": mse_loss.avg,
        "test/psnr":psnr.avg,
        "test/ssim":ssim.avg,

        
        }
        wandb.log(log_dict)


    else:

        print(
            f"valid epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"

        )
        log_dict = {
        "valid":epoch,
        "valid/loss": loss.avg,
        "valid/mse": mse_loss.avg,
        "valid/psnr":psnr.avg,
        "valid/ssim":ssim.avg,

        }  
        wandb.log(log_dict)
        if train_baseline is False:
            log_dict =  {
                "valid":epoch,
                "valid/accuracy":accuracy,
                "valid/crossentropy":gate_loss.avg,
                "valid/f1_score":f1
            }
            wandb.log(log_dict)

            wandb.log({
                "plot_valid":epoch,
                "plot_valid/conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=labels, preds=predictions,
                            class_names=["natural","sketch","clipart","painting"])})

    return loss.avg



def compress_with_ac_gate(model,  filelist, device, epoch,num_adapter, loop = True, name = "",  writing = None, oracles = None, train_baseline = False):
    #model.update(None, device)
    print("ho finito l'update")
    bpp_loss = AverageMeter()
    psnr_val = AverageMeter()
    mssim_val = AverageMeter()


    mean_softmax = torch.zeros((len(filelist),num_adapter)).to("cuda")
    
    with torch.no_grad():
        for i,d in enumerate(filelist): 

            x = read_image(d, adapt = False).to(device)


            if oracles is not None:
                orac = torch.tensor([oracles[i]]).to("cuda")
                print("oracles are: ",orac)
            else:
                orac = None

            nome_immagine = d.split("/")[-1].split(".")[0]
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)

            if train_baseline is False:
                data =  model.compress(x_padded)
                out_dec = model.decompress(data["strings"], data["shape"],data["gate_values"], oracle = orac)
            else:
                data =  model.compress(x_padded)
                out_dec = model.decompress(data["strings"], data["shape"])               

            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

            out_dec["x_hat"].clamp_(0.,1.)     

            if train_baseline is False:
                gate_probs = F.softmax(data["gate_values"])
                mean_softmax[i,:] = gate_probs




            psnr_im = compute_psnr(x, out_dec["x_hat"])
            ms_ssim_im = compute_msssim(x, out_dec["x_hat"])
            psnr_val.update(psnr_im)
            mssim_val.update(ms_ssim_im)
            
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]
            bpp = sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels#sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels #ssssss

            
            bpp_loss.update(bpp)

            if writing is not None:
                fls = writing + ".txt"
                f=open(fls , "a+")
                f.write("SEQUENCE "  +   nome_immagine + " BITS " +  str(bpp) + " PSNR " +  str(psnr_im)  + " MSSIM " +  str(ms_ssim_im) + "\n")
                f.close()  
            
            if "kodak" in name:
                print("image: ",d,": ",bpp," ",compute_psnr(x, out_dec["x_hat"]))



    if train_baseline is False:
        media = torch.mean(mean_softmax,dim = 0).detach().cpu()
        data = [s.item() for s in media]                 
        log_dict = {
                name + "compress":epoch,
                name + "compress/natural_distribution": data[0],
                name +  "compress/sketch_distribution":data[1],
                name +   "compress/clipart_distribution":data[2]   
        }
        wandb.log(log_dict)

    if loop:
        log_dict = {
                name + "compress":epoch,
               name + "compress/bpp_with_ac": bpp_loss.avg,
              name +  "compress/psnr_with_ac": psnr_val.avg,
             name +   "compress/mssim_with_ac":mssim_val.avg
        }
        
        wandb.log(log_dict)
        print("RESULTS OF COMPRESSION IS! : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    else:
        print("RESULTS OF COMPRESSION IS : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    
    if writing is not None:

        fls = writing + "_epochenc_" + str(epoch) +   ".txt"
        f=open(fls , "a+")
        f.write("SEQUENCE "  +   "AVG " + "BITS " +  str(bpp_loss.avg) + " YPSNR " +  str(psnr_val.avg)  + " YMSSIM " +  str(mssim_val.avg) + "\n")



    return psnr_val.avg, bpp_loss.avg







def evaluate_base_model_gate(model, args,device, epoch,num_adapter, oracle, train_baseline= False, writing = None):
    """
    Valuto la bontà del modello base, di modo da poter vedere se miglioraiamo qualcosa
    """

    model.to(device)
    model.update()

    test_transforms = transforms.Compose([transforms.ToTensor()])
    print("***************************************** KODAK *************************************************")
    kodak = AdapterDataset(root = args.root, path  =  ["test_kodak.txt"],classes = args.considered_classes, transform = test_transforms,train = False)
    kodak_f = kodak.samples
    kodak_filelist = []
    kodak_cl = [] if oracle else None
    for i in range(len(kodak_f)):
        kodak_filelist.append(kodak_f[i][0])
        if oracle:
            kodak_cl.append(int(kodak_f[i][1]))
    psnr, bpp = compress_with_ac_gate(model, 
                                      kodak_filelist, 
                                      device,
                                      epoch,
                                      num_adapter, 
                                      loop=True, 
                                      name= "kodak_", 
                                      oracles = kodak_cl,  
                                      train_baseline= train_baseline,
                                      writing = writing )
    print(psnr,"  ",bpp)

    print("****************************************** CLIC ****************************************************************")
    clic = AdapterDataset(root = args.root, path  =  ["test_clic.txt"],classes = args.considered_classes,transform = test_transforms,train = False)
    clic_f = clic.samples
    clic_filelist = []
    clic_cl = [] if oracle else None
    for i in range(len(clic_f)):
        clic_filelist.append(clic_f[i][0])
        if oracle:
            clic_cl.append(int(clic_f[i][1]))
    psnr, bpp = compress_with_ac_gate(model,
                                       clic_filelist,
                                         device,
                                         epoch,
                                         num_adapter,  
                                         loop=True,
                                           name = "clic_", 
                                           oracles = clic_cl,  
                                           train_baseline= train_baseline,
                                            writing = writing)
    print(psnr,"  ",bpp)
    """
    print("****************************************** sketch ****************************************************************")
    sketch = AdapterDataset(root = args.root, path  =  ["test_sketch.txt"], transform = test_transforms)
    sketch_f = sketch.samples
    sketch_filelist = []
    sketch_cl = [] if oracle else None
    for i in range(len(sketch_f)):
        sketch_filelist.append(sketch_f[i][0])
        if oracle:
            sketch_cl.append(int(sketch_f[i][1]))
    psnr, bpp = compress_with_ac_gate(model, 
                                      sketch_filelist, 
                                      device, 
                                      epoch, 
                                      num_adapter,

                                      loop=True, 
                                      name= "sketch_",
                                      oracles = sketch_cl, 
                                      train_baseline= train_baseline,
                                      writing = writing)
    print(psnr,"  ",bpp)
    """
    print("****************************************** clipart **************************************************************4444**")
    clipart = AdapterDataset(root = args.root, path  =  ["test_clipart.txt"],classes = args.considered_classes, transform = test_transforms,train = False)
    clipart_f = clipart.samples
    clipart_filelist = []
    clipart_cl = [] if oracle else None
    for i in range(len(clipart_f)):
        clipart_filelist.append(clipart_f[i][0])
        if oracle:
            clipart_cl.append(int(clipart_f[i][1]))
    psnr, bpp = compress_with_ac_gate(model, 
                                      clipart_filelist, 
                                      device, 
                                      epoch, 
                                      num_adapter,
                                      loop=True, 
                                      name =  "clipart_", 
                                      oracles= clipart_cl, 
                                      train_baseline= train_baseline,
                                       writing = writing)
    print(psnr,"  ",bpp)

    print("****************************************** PAINTING***************************************+++*************************")
    painting = AdapterDataset(root = args.root, path  =  ["test_painting.txt"],classes = args.considered_classes, transform = test_transforms,train = False)
    painting_f = painting.samples
    painting_filelist = []
    painting_cl = [] if oracle else None
    for i in range(len(painting_f)):
        painting_filelist.append(painting_f[i][0])
        if oracle:
            painting_cl.append(int(painting_f[i][1]))
    psnr, bpp = compress_with_ac_gate(model, 
                                      painting_filelist, 
                                      device, 
                                      epoch, 
                                      num_adapter,
                                      loop=True, 
                                      name =  "painting_", 
                                      oracles= painting_cl, 
                                      train_baseline= train_baseline,
                                       writing = writing)
    print(psnr,"  ",bpp)