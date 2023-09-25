
import torch 
import math
import torch.nn as nn
from compress.ops import ste_round

class RateDistortionLoss(nn.Module):

    def __init__(self, lmbda = 1e-2):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = lmbda 
        self.type = type


    def forward(self, output, target):
        N, _, H, W = target.size()      
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] 

        return out  
    



class AdapterLoss(nn.Module):
    """
    Loss function just to train the adapter, the loss is the MAE between teacher quantizer and student quantization
    """
    def __init__(self, teacher_model):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.teacher_model = teacher_model



    

    def forward(self,output, target):





        out = {}

        N, _, H, W = target.size()      
        
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   

        y_teacher = self.teacher_model.encode_latent(out["x"])


        out["loss"] = self.dist_metric(out["y_hat"], y_teacher) 

        return out




class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, teacher_model, lmbda = 0):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = lmbda 
        self.teacher_model = teacher_model


    def forward(self, output, target):
    
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)       
       




        N, _, H, W = target.size()      
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   



        y_teacher = self.teacher_model.encode_latent(target)


        out["adapter_loss"] = self.dist_metric(output["y_hat"], y_teacher) 

        #print("************************************************ ",out["adapter_loss"],"  ",255*out["mse_loss"])


        out["loss"] =   (1 - self.lmbda)  * 255 *out["mse_loss"]  +  self.lmbda*out["adapter_loss"]


        return out




class DistorsionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = 0.0483



    def forward(self, output, target):
    
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)       
        out["loss"] =   self.lmbda * 255 ** 2 *out["mse_loss"]




        N, _, H, W = target.size()      
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   



        return out
    
class RateLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dist_metric = nn.MSELoss()




    def forward(self, output, target):
    
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)       
        




        N, _, H, W = target.size()      
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   

        out["loss"] =   out["bpp_loss"]



        return out




    


# modified from https://github.com/InterDigitalInc/CompressAI/blob/master/examples/train.py
# Copyright (c) 2021-2022 InterDigital Communications, Inc Licensed under BSD 3-Clause Clear License.
class RateDistortionModelLoss(nn.Module):
    # modified from RateDistortionLoss
    def __init__(self, lmbda: float = 1e-2) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output: dict, target: torch.Tensor) -> dict:
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = 0

        for likelihood in output["likelihoods"].values():
            if likelihood is not None:
                out["bpp_loss"] += torch.log(likelihood).sum() / (
                    -math.log(2) * num_pixels
                )

        for likelihood in output["m_likelihoods"].values():
            if likelihood is not None:
                out["bpp_loss"] += likelihood.log().sum() / (-math.log(2) * num_pixels)

        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out








