
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
    def __init__(self, quantization_policy = None):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.quantization_policy = quantization_policy
        if self.quantization_policy is not None:
            self.w = quantization_policy["w"]
            self.length = self.w.shape[0]
            self.cum_w = self.compute_cumulative_weights()
            self.sym_w = self.compute_symmetric_weights()
            if "b" in list(self.quantization_policy.keys()):
                self.b = self.quantization_policy["b"] #self.compute_bias()
            else: 
                self.b = self.compute_bias()
        else: 
            self.w = None

    def compute_cumulative_weights(self):
        self.cum_w = torch.zeros(self.length + 1)
        self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
        self.cum_w = torch.cat((-torch.flip(self.cum_w[1:], dims = [0]),self.cum_w),dim = 0)
        


    def compute_symmetric_weights(self):
        return torch.cat((torch.flip(self.w,[0]),self.w),0) 

    def compute_bias(self):
        return  torch.add(self.cum_w[1:], self.cum_w[:-1])/2


    

    def forward(self,output, target):


        y_teacher = output["y_teacher"]
        y_student = output["y_hat"]
        
        # flattent everything
        #y = y.flatten()[None,None,:]hh
        #y_student = y_student.flatten() #[None,None,:]
        #quantize y_teacher with teacher quantization module 
        #if self.w is None: 
        #y_teacher = torch.round(y)
        #else: 
        #y_teacher = self.quantize(y)
        #y_teacher = y_teacher.flatten()




        out = {}

        N, _, H, W = target.size()      
        
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   

 
        out["loss"] = self.dist_metric(y_teacher, y_student)*255**2/3

        return out




class DistorsionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dist_metric = nn.MSELoss()



    def forward(self, output, target):
    
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)       
        out["loss"] =   out["mse_loss"]




        N, _, H, W = target.size()      
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   



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








