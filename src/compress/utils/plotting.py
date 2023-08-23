
import torch 
import wandb









def plot_sos(model, device,n = 10000, dim = 0,):



    x_min = float((min(model.gaussian_conditional.sos.b) + min(model.gaussian_conditional.sos.b)*0.5).detach().cpu().numpy())
    x_max = float((max(model.gaussian_conditional.sos.b)+ max(model.gaussian_conditional.sos.b)*0.5).detach().cpu().numpy())
    step = (x_max-x_min)/n
    x_values = torch.arange(x_min, x_max, step)
    x_values = x_values.repeat(model.gaussian_conditional.M,1,1)
            
    y_values= model.gaussian_conditional.sos(x_values.to(device), mode = "dequantize")[0,0,:]
    data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
    table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
    wandb.log({"GaussianSoS/Gaussian SoS  inf at dimension " + str(dim): wandb.plot.line(table_inf, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(-1))})  


