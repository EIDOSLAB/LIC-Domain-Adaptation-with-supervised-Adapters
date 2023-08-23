
import torch 
import math
import torch.nn as nn


class RateDistortionLossQuantizationError(nn.Module):

    def __init__(self,lmbda = 1e-2, gamma = 1e-2):

        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.quantization_metric = nn.L1Loss()
        self.lmbda = lmbda 
        self.type = type
        self.gamma = gamma


    def forward(self, output, target):
        N, _, H, W = target.size()      
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        
        out["quantization_loss"] = self.quantization_metric(output["y_hat"],output["y"])

        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]  + self.gamma*out["quantization_loss"]

        return out  
    









