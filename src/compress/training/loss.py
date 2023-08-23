
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
    





    








class ControlledRDLoss(nn.Module):

    def __init__(self, lmbda = 1, norm_term = 1e-2, target_psnr = 35.0):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = lmbda 
        self.norm_term = norm_term
        self.target_psnr = target_psnr 

        self.target_mse = (255**2)/(10**(self.target_psnr/10))
        self.target_mse = self.target_mse*self.norm_term

        print("stampo il target mse -----> ",self.target_mse)


    def forward(self, output, target):
        N, _, H, W = target.size()      
        out = {}
        out["mse_loss"] = 255**2 *self.dist_metric(output["x_hat"], target)*self.norm_term
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        



        out["loss"] = self.lmbda * torch.abs( out["mse_loss"] - self.target_mse) + out["bpp_loss"] 

        return out  
    




def r_squared(y_true, y_pred):
    # Calcola la media del target reale
    y_mean = torch.mean(y_true)

    # Calcola la somma dei quadrati totali (SST)
    sst = torch.sum((y_true - y_mean) ** 2)

    # Calcola la somma dei quadrati dei residui (SSE)
    sse = torch.sum((y_true - y_pred) ** 2)

    # Calcola il coefficiente di determinazione R²
    r_squared = 1 - (sse / sst)
    return r_squared



class ResidualLoss(nn.Module):

    def __init__(self, lmbda = 1e-2, loss = "mse"):
        super().__init__()

        self.dist_metric = nn.MSELoss()
        self.lmbda = lmbda 
        self.loss = loss


        if self.loss == "mse": 
            self.res_metric = nn.MSELoss()
        elif self.loss == "mae":
            self.res_metric = nn.MAELoss()
        else:
            self.res_metric == r_squared()



    def forward(self, output, target):
        N, _, H, W = target.size()      
        out = {}
        out["mse_loss"] = self.dist_metric(output["x_hat"], target)
        num_pixels = N * H * W
    
           


        out["y_bpp"] = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        out["z_bpp"]  = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels) 
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values())   
        

        out["res_met"] = self.res_metric(output["residual"][0], output["residual"][1])



        
        out["bpp_res"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in output["res_likelihoods"].values())

        #print("------------------------------->    ",out["res_met"],"    ",out["bpp_res"])

        out["loss"] = self.lmbda*out["res_met"] + out["bpp_res"] # self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] 

        return out 