
import torch 
import math
import torch.nn as nn



class GateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.dist_metric = nn.MSELoss()

    def forward(self,output, target):
        out = {}
        out["CrossEntropy"] = self.loss(output["logits"], target[1]) 
        out["loss"] = out["CrossEntropy"]
        out["mse_loss"] = self.dist_metric(output["x_hat"], target[0])

        return out
    


class GateDistorsionLoss(nn.Module):
    def __init__(self, lmbda = 0.1):
        super().__init__()
        self.lmbda = lmbda 
        self.gate_metric = nn.CrossEntropyLoss()
        self.dist_metric = nn.MSELoss()




    def forward(self, output, target):

        out = {}

        logits = output["logits"]
        classes = target[1]
        out["CrossEntropy"] = self.gate_metric(logits, classes)
        N, _, H, W = target[0].size()      

        
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        
        
        out["mse_loss"] = self.lmbda * 255**2 *self.dist_metric(output["x_hat"], target[0])  # lambda è qua
        out["loss"] =   out["mse_loss"] + out["CrossEntropy"]

        return out  
    



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




    



