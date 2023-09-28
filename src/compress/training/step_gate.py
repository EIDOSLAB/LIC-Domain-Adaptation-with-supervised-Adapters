import torch 

import wandb
from compress.utils.help_function import compute_msssim, compute_psnr
import torch.nn.functional as F 
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import math
from sklearn.metrics import confusion_matrix
from compressai.ops import compute_padding

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


    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/crossentropy":gate_loss.avg

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
        
        }
        wandb.log(log_dict)
        #wandb.log({"Confusion Matrix Epoch  " + str(epoch) : wandb.plot.confusion_matrix(cm, class_names=3)})


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
        
        }  

        wandb.log(log_dict)
        #wandb.log({"Confusion Matrix Epoch " + str(epoch): wandb.plot.confusion_matrix(cm, class_names=3)})     

    
   

    return loss.avg