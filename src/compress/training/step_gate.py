import torch 

import wandb
from compress.utils.help_function import compute_msssim, compute_psnr
import torch.nn.functional as F 
from PIL import Image
from torchvision import transforms
import math
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


def train_one_epoch_gate(model, criterion, train_dataloader, optimizer,training_policy, epoch, counter):


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
        #if aux_optimizer is not None:
        #    aux_optimizer.zero_grad()


        if training_policy != "gate":
            out_net = model(d)
        else:
             out_net = model.forward_gate(d)
        out_criterion = criterion(out_net, (d,cl))

        out_criterion["loss"].backward()
        optimizer.step()

        loss.update(out_criterion["loss"].clone().detach())
        if training_policy != "gate":
            mse_loss.update(out_criterion["mse_loss"].clone().detach())
            bpp_loss.update(out_criterion["bpp_loss"].clone().detach())

        gate_loss.update(out_criterion["CrossEntropy"].clone().detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)


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

        wandb_dict = {
            "train_batch":counter,
            "train_batch/losses_batch":out_criterion["loss"].clone().detach().item(),
            "train_batch/crossentropy":out_criterion["CrossEntropy"].clone().detach().item(),
        }

        wandb.log(wandb_dict)

        if training_policy != "gate":

            wand_dict = {
                "train_batch": counter,
                "train_batch/mse":out_criterion["mse_loss"].clone().detach().item(),

            }
            wandb.log(wand_dict)

    accuracy = 100 * correct / total

    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/crossentropy":gate_loss.avg,
        "train/accuracy":accuracy

    }


    if training_policy != "gate":

        log_dict = {
            "train":epoch,
            "train/mse": mse_loss.avg,
            }
            
        wandb.log(log_dict)
    return counter

import numpy as np



def test_epoch_gate(epoch, test_dataloader, model, criterion, training_policy, valid):
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
            d = F.pad(d, pad, mode="constant", value=0)


            if training_policy != "gate":
                out_net = model(d)
            else:
                out_net = model.forward_gate(d)
            


            out_criterion = criterion(out_net, (d,cl))

            out_net["x_hat"] = F.pad(out_net["x_hat"], unpad)
            #if valid:
            _, predicted = torch.max(F.softmax(out_net["logits"]), 1) # predicted
            #else:
            #    _, predicted = torch.max(F.softmax(out_net["logits"]))


            predictions.extend(predicted.cpu().numpy().tolist())
            labels.extend(cl.cpu().numpy().tolist())

            total += cl.size(0)
            correct += (predicted == cl).sum().item()
            
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"] + 0.00001)

            if training_policy != "gate":
                psnr.update(compute_psnr(d, out_net["x_hat"]))
                ssim.update(compute_msssim(d, out_net["x_hat"]))
            
            gate_loss.update(out_criterion["CrossEntropy"])


    accuracy = 100 * correct / total
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


        log_dict = {
        "test":epoch,
        "test/loss": loss.avg,
        "test/mse": mse_loss.avg,
        "test/psnr":psnr.avg,
        "test/ssim":ssim.avg,
        "test/accuracy":accuracy,
        "test/crossentropy":gate_loss.avg,
        "test/f1_score":f1
        
        }
        wandb.log(log_dict)
        wandb.log({
            "plot_test":epoch,
            "plot_test/conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=labels, preds=predictions,
                        class_names=["natural","sketch","clipart"])})


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
        "valid/accurcay":accuracy,
        "valid/crossentropy":gate_loss.avg,
        "valid/f1_score":f1
        
        }  

        wandb.log(log_dict)
        wandb.log({
            "plot_valid":epoch,
            "plot_valid/conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=labels, preds=predictions,
                        class_names=["natural","sketch","clipart"])})



    
   

    return loss.avg



def compress_with_ac_gate(model,  filelist, device, epoch, loop = True,  writing = None ):
    #model.update(None, device)
    print("ho finito l'update")
    bpp_loss = AverageMeter()
    psnr_val = AverageMeter()
    mssim_val = AverageMeter()

    
    with torch.no_grad():
        for i,d in enumerate(filelist): 

            x = read_image(d, adapt = False).to(device)


            nome_immagine = d.split("/")[-1].split(".")[0]
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)


            data =  model.compress(x_padded)
            out_dec = model.decompress(data["strings"], data["shape"],data["gate_values"])

            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

            out_dec["x_hat"].clamp_(0.,1.)     


            # parte la prova, devo controllare che y_hat sia sempre uguale!!!!! 
            #out = model.forward(x)
            #y_hat_t = out["y_hat"].ravel()
            #y_hat_comp = out_dec["y_hat"].ravel()
            
            #for i in range(10):
            #    print("----> ", y_hat_t[i]," ",y_hat_comp[i],"  ",out_dec["x_hat"].shape)





            psnr_im = compute_psnr(x, out_dec["x_hat"])
            ms_ssim_im = compute_msssim(x, out_dec["x_hat"])
            psnr_val.update(psnr_im)
            mssim_val.update(ms_ssim_im)
            
            size = out_dec['x_hat'].size()
            num_pixels = size[0] * size[2] * size[3]
            bpp = sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels#sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels

            
            bpp_loss.update(bpp)

            if writing is not None:
                fls = writing + ".txt"
                f=open(fls , "a+")
                f.write("SEQUENCE "  +   nome_immagine + " BITS " +  str(bpp) + " PSNR " +  str(psnr_im)  + " MSSIM " +  str(ms_ssim_im) + "\n")
                f.close()  
                
            #print("image: ",d,": ",bpp," ",compute_psnr(x, out_dec["x_hat"]))




                    

    if loop:
        log_dict = {
                "compress":epoch,
                "compress/bpp_with_ac": bpp_loss.avg,
                "compress/psnr_with_ac": psnr_val.avg,
                "compress/mssim_with_ac":mssim_val.avg
        }
        
        wandb.log(log_dict)
        print("RESULTS OF COMPRESSION IS : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    else:
        print("RESULTS OF COMPRESSION IS : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    
    if writing is not None:

        fls = writing + ".txt"
        f=open(fls , "a+")
        f.write("SEQUENCE "  +   "AVG " + "BITS " +  str(bpp_loss.avg) + " YPSNR " +  str(psnr_val.avg)  + " YMSSIM " +  str(mssim_val.avg) + "\n")



    return psnr_val.avg, bpp_loss.avg



def bpp_calculation(out_net, out_enc, fact = False):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]
        if fact is False:
            bpp_1 = (len(out_enc[0]) * 8.0 ) / num_pixels
            bpp_2 =  (len(out_enc[1]) * 8.0 ) / num_pixels
            return bpp_1 + bpp_2, bpp_1, bpp_2
        else:
            bpp = (len(out_enc[0]) * 8.0 ) / num_pixels 
            return bpp, bpp, bpp


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:


    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    print("la metrica psnr è questa ", metrics["psnr"])
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    print("la metrica psnr è questa ", metrics["psnr"],"   ",metrics["ms-ssim"])
    return metrics
