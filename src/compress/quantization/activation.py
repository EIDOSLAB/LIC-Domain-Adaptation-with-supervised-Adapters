
import torch
import torch.nn as nn

      

class AdaptiveQuantization(nn.Module):
    def __init__(self, extrema, trainable , device = torch.device("cuda")):
        super().__init__()
        self.extrema = extrema
        self.trainable = trainable 
        self.minimo = -extrema  
        self.massimo = extrema         
        self.range_num = torch.arange(0.5 ,self.massimo ).type(torch.FloatTensor)
        self.w = torch.nn.Parameter(torch.ones(len(self.range_num)), requires_grad= trainable)
        self.length = len(self.range_num)

        self.map_sos_cdf = {}
        self.map_cdf_sos = {}
        
        self.update_state(device = device)




    def update_weights(self, device = torch.device("cuda")):
        self.sym_w =  torch.cat((torch.flip(self.w,[0]),self.w),0).to(device)
        self.b = self.average_points.to(device)





    def update_state(self, device = torch.device("cuda")):
        
        self.update_cumulative_weights(device)
        self.calculate_average_points(device) #self.average_points
        self.calculate_distance_points(device) #self.distance_points
        self.update_weights(device)
        self.define_channels_map()
        


    def update_cumulative_weights(self,device):
        self.cum_w = torch.zeros(self.length + 1)
        self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
        self.cum_w = torch.cat((-torch.flip(self.cum_w[1:], dims = [0]),self.cum_w),dim = 0)
        self.cum_w = self.cum_w.to(device)

    
    def calculate_average_points(self,device):
        self.average_points =  torch.add(self.cum_w[1:], self.cum_w[:-1])/2
        self.average_points = self.average_points.to(device)


    def calculate_distance_points(self,device):
        self.distance_points =   torch.sub(self.cum_w[1:], self.cum_w[:-1])/2
        self.distance_points = self.distance_points.to(device)


    def define_channels_map(self ):

        mapping = torch.arange(0, int(self.cum_w.shape[0]), 1).numpy()
        map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))
        map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            
        self.map_sos_cdf = map_float_to_int
        self.map_cdf_sos = map_int_to_float


    def define_v0_and_v1(self, inputs): 


        inputs_shape = inputs.shape
        inputs = inputs.reshape(-1) #.to(inputs.device) # perform reshaping 
        inputs = inputs.unsqueeze(1)#.to(inputs.device) # add a dimension
       
        average_points = self.average_points#.to(inputs.device)
        distance_points = self.distance_points#.to(inputs.device)
       
        
        leftest_points =  self.cum_w[0] - distance_points[0]
        rightest_points = self.cum_w[-1] + distance_points[-1]


        average_points_left = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) - leftest_points # 1000 è messo a caso al momento 
        average_points_left[1:] = average_points
        average_points_left = average_points_left.unsqueeze(0)#.to(inputs.device)
        

        average_points_right = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) + rightest_points # 1000 è messo a caso al momento 
        average_points_right[:-1] = average_points
        average_points_right = average_points_right.unsqueeze(0)#.to(inputs.device)       
               
               
        distance_points_left = torch.cat((torch.tensor([0]).to(inputs.device),distance_points),dim = -1).to(inputs.device)
        distance_points_left = distance_points_left.unsqueeze(0)#.to(inputs.device)
        
        distance_points_right = torch.cat((distance_points, torch.tensor([0]).to(inputs.device)),dim = -1).to(inputs.device)
        distance_points_right = distance_points_right.unsqueeze(0)#.to(inputs.device)
        
        li_matrix = inputs > average_points_left # 1 if x in inputs is greater that average point, 0 otherwise. shape [__,15]
        ri_matrix = inputs <= average_points_right # 1 if x in inputs is smaller or equal that average point, 0 otherwise. shape [__,15]
        
        #li_matrix = li_matrix.to(inputs.device)
        #ri_matrix = ri_matrix.to(inputs.device)

        one_hot_inputs = torch.logical_and(li_matrix, ri_matrix).to(inputs.device) # tensr that represents onehot encoding of inouts tensor (1 if in the interval, 0 otherwise)
              
        one_hot_inputs_left = torch.sum(distance_points_left*one_hot_inputs, dim = 1).unsqueeze(1)#.to(inputs.device) #[1200,1]
        one_hot_inputs_right = torch.sum(distance_points_right*one_hot_inputs, dim = 1).unsqueeze(1)#.to(inputs.device) #[1200,1]
        
        
        v0 = one_hot_inputs_left.reshape(inputs_shape)#.to(inputs.device) #  in ogni punto c'è la distanza con il livello a sinistra       
        v1 = one_hot_inputs_right.reshape(inputs_shape)#.to(inputs.device) # in ogni punto c'è la distanza con il livello di destra

        return v0 , v1




    def forward(self, x, mode):
        """
        in this step we quantize adaptively the latent representation 
        1 - take x 
        if mode == "training" ----> xt = x + unif(li,ri)
        mode == "dequantize" ------> actual quantization 
        mode == "encoding" -------> actual quantization + mapping (implemento dopo)
        """
        #print("PRIMMMMMMMMMMMMMMMA ",x.shape)
        v0,v1 = self.define_v0_and_v1(x)
        #print("***************************************")
        #print(torch.max(v0)," ",torch.min(v0))
        #print(torch.max(v1)," ",torch.min(v1))
        li = x - v0 # tensore con valori a sinistra x - li
        ri = x + v1 # tensore con valori estremi a destra x + ri
        if mode == "training": 
            """
            half = float(0.5)
            noise = torch.empty_like(x).uniform_(-half, half)
            inputs = x + noise
            return inputs 
            """
            noise = torch.empty_like(x).uniform_()
            output = (li -ri)*noise + ri
            return output
        else:
            #return torch.stack([w[i]*(torch.relu(torch.sign(x-b[i]))) - w[i]/2 for i in range(self.length)], dim=0).sum(dim=0) 
            c = torch.sum((self.sym_w[:,None]/2)*(torch.sign(x - self.b[:,None])),dim = 1).unsqueeze(1)
            return c
