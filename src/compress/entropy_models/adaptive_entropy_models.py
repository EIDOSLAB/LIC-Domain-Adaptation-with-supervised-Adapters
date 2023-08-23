
import torch.nn as nn 
import torch 
import numpy as np

from typing import Any, Callable,  Optional, Tuple
from torch import Tensor
from compressai.ops import LowerBound

from compress.quantization.activation import AdaptiveQuantization
import torchac
from compress.entropy_models.coder import _EntropyCoder, default_entropy_coder, _forward
import torch.nn.functional as F
import copy
from compress.entropy_models.coder import _EntropyCoder, default_entropy_coder, pmf_to_quantized_cdf, _forward


def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])


class AdaptedEntropyModel(nn.Module):
    r"""Entropy model base class.
    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(self,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)
        
        
        self.interval_min_value = []
        self.interval_max_value = []

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        #self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length


    forward: Callable[..., Any] = _forward



    def transform_float_to_int(self,x):
        if x not in self.sos.unique_values:
            raise ValueError("the actual values ",x," is not present in ",self.sos.cum_w)
        return int((self.sos.unique_values ==x).nonzero(as_tuple=True)[0].item())
    

    def transform_int_to_float(self,x):
        return self.sos.unique_values[x].item()


    def transform_map(self,x,map_float_to_int):
        if x in map_float_to_int.keys():
            return map_float_to_int[x]
        else:
            # find the closest key and use this
            keys = np.asarray(list(map_float_to_int.keys()))
            keys = torch.from_numpy(keys).to(x.device)
            i = (torch.abs(keys - x)).argmin()
            key = keys[i].item()
            return map_float_to_int[key]

    def define_permutation(self, x):
        perm = np.arange(len(x.shape)) 
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)] # perm and inv perm
        return perm, inv_perm




    def quantize(self, inputs, mode,  means = None, permutation = False):



        if permutation is True: 
            perm, inv_perm = self.define_permutation(inputs)
            perms = [perm,inv_perm]
        else: 
            perms = None

        #print("si parte da qua: ",inputs.shape)
        if permutation is True:
            inputs =  inputs.permute(*perms[0]).contiguous() # flatten y and call it values
            shape = inputs.size() 
            inputs = inputs.reshape(1, 1, -1) # reshape values
            if means is not None:
                means = means.permute(*perms[0]).contiguous()
                means = means.reshape(1, 1, -1).to(inputs.device)     


        if mode == "training":
            outputs = self.sos(inputs, mode)
            if perms is True:
                outputs =outputs.reshape(shape)
                outputs = outputs.permute(*perms[1]).contiguous()
            
            return outputs
            
        
        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = self.sos( outputs, mode)  

        if mode == "dequantize":
            if means is not None:
                outputs += means

            if permutation is True:
                outputs =outputs.reshape(shape)
                outputs = outputs.permute(*perms[1]).contiguous()
            return outputs

        if permutation is True:
            outputs =outputs.reshape(shape)
            outputs = outputs.permute(*perms[1]).contiguous()


        assert mode == "symbols", mode
        shape_out = outputs.shape
        outputs = outputs.ravel()
        map_float_to_int = self.sos.map_sos_cdf 
        
        for i in range(outputs.shape[0]):
            outputs[i] =  self.transform_map(outputs[i], map_float_to_int)

        outputs = outputs.reshape(shape_out)    
        return outputs
        

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf


     
    def dequantize(self, inputs, means = None):
        """
        we have to 
        1 -map again the integer values to the real values for each channel
        2 - ad the means  
        """
        inputs = inputs.to(torch.float)
        map_int_to_float = self.sos.map_cdf_sos
        shape_inp = inputs.shape
        inputs = inputs.ravel()
        for i in range(inputs.shape[0]):
            c = torch.tensor(map_int_to_float[inputs[i].item()],dtype=torch.float32)
            inputs[i] = c.item()
        #for i in range(self.M):            
        #    inputs[0,i,:,:] = inputs[0,i,:,:].apply_(lambda x: self.sos.map_cdf_sos[i][x])
        inputs = inputs.reshape(shape_inp)
        if means is not None:
            inputs += means        
        outputs = inputs.type(torch.float)
        #print(" factorized terzo utputs da confrontare: ",torch.unique(outputs[0,:,:], return_counts=True))
        return outputs



            
            
    def extract_cdf(self, inp_dix):
        res = torch.zeros(self.cdf.shape) 
        for i in range(self.M):
            tmp = inp_dix[i,:] # pmf
            tmp =  self.pmf_to_cdf(tmp) # cdf 
            res[i,:] = tmp 
        return res
        



    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])



    def compress(self, inputs):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        indexes = self._build_indexes(inputs.size())
        symbols = self.quantize(inputs, "symbols", permutation = True) 

        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputsxxx` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        #self._check_cdf_size()
        #self._check_cdf_length()
        #self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(), # da settare 
                self._cdf_length.reshape(-1).int().tolist(),  # set 
                self._offset.reshape(-1).int().tolist(),  # offset is 0 
            )
            strings.append(rv)


        return strings




    def decompress(self, strings, indexes):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())


        outputs = self.dequantize(outputs)
        return outputs



    def compress_torcach(self, inputs ):
        symbols = inputs #[1]
        M = symbols.size(1)
        #print("M  per il bottleneck is: ",M)
        symbols = symbols.to(torch.int16)      
        output_cdf = torch.zeros_like(symbols,dtype=torch.int16).to(inputs.device)
        output_cdf = output_cdf[:,:,:,:,None].to(inputs.device) + torch.zeros(self.cdf.shape[1]).to(inputs.device)
        #print("output cdf shape: ",output_cdf.shape)
        for i in range(M):
            output_cdf[:,i,:,:,:] = self.cdf[i,:]         
        # ci si muove nella cpu 
        output_cdf = output_cdf.to("cpu")
        symbols = symbols.to("cpu")   
        byte_stream = torchac.encode_float_cdf(output_cdf, symbols, check_input_bounds=True)      

        if torchac.decode_float_cdf(output_cdf, byte_stream).equal(symbols) is False:
            raise ValueError("il simbolo codificato è different, qualcosa non va!")
        return byte_stream, output_cdf



    def decompress_torcach(self, byte_stream, output_cdf):
        output = torchac.decode_float_cdf(output_cdf, byte_stream)

        return output


class AdaptedEntropyBottleneck(AdaptedEntropyModel):

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        pretrained_entropy_model = None,
        trainable: bool  = True,        
        extrema: int = 10,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        device = torch.device("cuda"),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.trainable = trainable 
        self.M = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.extrema = extrema
        self.pmf = None

        self.pmf_length = None
        

        self.sos = AdaptiveQuantization(extrema = self.extrema, trainable = self.trainable, device = device)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.M
        num_params = 0
        for i in range(len(self.filters) + 1):
            if pretrained_entropy_model is None:
                init = np.log(np.expm1(1 / scale / filters[i + 1]))
                matrix = torch.Tensor(channels, filters[i + 1], filters[i])
                num_params += matrix.reshape(-1).shape[0]
                matrix.data.fill_(init)
                self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))
                # mormal bias initialization 
                bias = torch.Tensor(channels, filters[i + 1], 1)
                num_params += bias.reshape(-1).shape[0]
                nn.init.uniform_(bias, -0.5, 0.5)
                self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))
                # normal filter initialization
                if i < len(self.filters):
                    factor = torch.Tensor(channels, filters[i + 1], 1)
                    num_params += factor.reshape(-1).shape[0]
                    nn.init.zeros_(factor)
                    self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))             
            else:
                
                # pretrained weights initialization 
                init_matrix = getattr(pretrained_entropy_model, f"_matrix{i:d}").data
                matrix = copy.deepcopy(init_matrix) #torch.Tensor(channels, filters[i + 1], filters[i]) 
                num_params += matrix.reshape(-1).shape[0]
                self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))
                # pretrianed bias initialization
                init_bias = getattr(pretrained_entropy_model, f"_bias{i:d}").data
                bias = copy.deepcopy(init_bias)
                num_params += bias.reshape(-1).shape[0]
                self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))
                
                if i < len(self.filters):
                    init_factor = factor = getattr(pretrained_entropy_model, f"_factor{i:d}")
                    factor = copy.deepcopy(init_factor)
                    num_params += factor.reshape(-1).shape[0]
                    self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))
        print(" total number of parameters for the entropy model: ",num_params)      
                
            
        target = np.log(2 / self.tail_mass - 1)
        self.target = torch.Tensor([-target, 0, target])
        #self.register_buffer("target", torch.Tensor([-target, 0, target]))
        
        


    def freeze_entropy_model(self):

        """
        Serve se voglio reinizializzare da capo il mio modello entropico!
        """
        print("freezing THE ENTROPY MODEL")

        for i in range(len(self.filters) + 1):          
            #matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix = getattr(self, f"_matrix{i:d}")
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix, requires_grad=False))
            bias = getattr(self, f"_bias{i:d}")
            bias = bias.detach()
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias, requires_grad=False))
            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor, requires_grad=False)) 





    def initialize_entropy_model(self, entropy_model = None):
        """
        Serve se voglio reinizializzare da capo il mio modello entropico!
        """
        print("REINITIALIZING THE ENTROPY MODEL")

        if entropy_model is None:
            filters = (1,) + self.filters + (1,)
            scale = self.init_scale ** (1 / (len(self.filters) + 1))
            for i in range(len(self.filters) + 1):
                init = np.log(np.expm1(1 / scale / filters[i + 1]))
                #matrix = torch.Tensor(channels, filters[i + 1], filters[i])
                matrix = getattr(self, f"_matrix{i:d}")
                matrix.data.fill_(init)
                self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

                bias = getattr(self, f"_bias{i:d}")
                nn.init.uniform_(bias, -0.5, 0.5)
                self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

                if i < len(self.filters):
                    factor = getattr(self, f"_factor{i:d}")
                    nn.init.zeros_(factor)
                    self.register_parameter(f"_factor{i:d}", nn.Parameter(factor)) 
        else:
            print("devo entrare qua!")
            filters = (1,) + self.filters + (1,)
            scale = self.init_scale ** (1 / (len(self.filters) + 1))
            for i in range(len(self.filters) + 1):

                #matrix = torch.Tensor(channels, filters[i + 1], filters[i])
                matrix = getattr(self, f"_matrix{i:d}")
                self.register_parameter(f"_matrix{i:d}", nn.Parameter(getattr(entropy_model.entropy_bottleneck,f"_matrix{i:d}").data))

                bias = getattr(self, f"_bias{i:d}")

                self.register_parameter(f"_bias{i:d}", nn.Parameter(getattr(entropy_model.entropy_bottleneck, f"_bias{i:d}").data))

                if i < len(self.filters):
                    factor = getattr(self, f"_factor{i:d}")
                    self.register_parameter(f"_factor{i:d}", nn.Parameter( getattr(entropy_model.entropy_bottleneck, f"_factor{i:d}").data)) 


    
    def find_closest_key(self, dic, val):
        keys = torch.tensor(list(dic.keys()))
        i = (torch.abs(keys - val.item())).argmin()
        key = keys[i]
        return key.item()


    def pmf_to_cdf(self, prob_tens = None):
        if prob_tens is None:
            cdf = self.pmf.cumsum(dim=-1)
            spatial_dimensions = self.pmf.shape[:-1] + (1,)
            zeros = torch.zeros(spatial_dimensions, dtype=self.pmf.dtype, device=self.pmf.device)
            cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
            cdf_with_0 = cdf_with_0.clamp(max=1.)
            return cdf_with_0
        else:
            cdf = prob_tens.cumsum(dim= -1)
            cdf_with_0 = torch.zeros(cdf.shape[0] + 1)
            cdf_with_0[1:] =  cdf
            return cdf_with_0            




    def update(self, device = torch.device("cuda")):

        self.sos.update_state(device) 
 


        samples = self.sos.cum_w
        

        samples = samples.repeat(self.M,1).unsqueeze(1) 
        samples = samples.to(device)
        # calculate the right intervals
        average_points = self.sos.average_points
        distance_points = self.sos.distance_points


        pmf_length = torch.zeros(self.M).to(device) + self.sos.cum_w.shape[0] 
        pmf_length = pmf_length.int()
        max_length = pmf_length.max().item()

        
        average_points.to(device)
        distance_points.to(device)
        v0,v1 = self.define_v0_and_v1(samples,average_points, distance_points)
       
        v0 = v0.to(device)
        v1 = v1.to(device)
        lower = self._logits_cumulative(v0, stop_gradient=True)
        upper = self._logits_cumulative(v1, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]  
        
        
        self._offset = torch.zeros(self.M).to(device)
        self.pmf = pmf
        self.cdf = self.pmf_to_cdf()



        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])
        

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2


        return True
    





        
    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool):
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")               
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits 
    





    
    def define_v0_and_v1(self, inputs, average_points, distance_points): 
        a,_,b = inputs.shape  # save the shape 
        inputs = inputs.reshape(-1)#.to(inputs.device) # perform reshaping 
        inputs = inputs.unsqueeze(1)#.to(inputs.device) # add a dimension
        
        average_points = average_points.to(inputs.device)
        distance_points = distance_points.to(inputs.device)
       

        leftest_points =  self.sos.cum_w[0] - distance_points[0]
        rightest_points = self.sos.cum_w[-1] + distance_points[-1]
        
        average_points_left = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) - leftest_points # 1000 è messo a caso al momento 
        average_points_left[1:] = average_points
        average_points_left = average_points_left.unsqueeze(0)#.to(inputs.device)
        

        average_points_right = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) + rightest_points # 1000 è messo a caso al momento 
        average_points_right[:-1] = average_points
        average_points_right = average_points_right.unsqueeze(0)#.to(inputs.device)       
               
        average_points = average_points.unsqueeze(0)#.to(inputs.device) # move in thanh class 
               
        distance_points_left = torch.cat((torch.tensor([0]).to(inputs.device),distance_points),dim = -1).to(inputs.device)
        distance_points_left = distance_points_left.unsqueeze(0)#.to(inputs.device)
        
        distance_points_right = torch.cat((distance_points, torch.tensor([0]).to(inputs.device)),dim = -1).to(inputs.device)
        distance_points_right = distance_points_right.unsqueeze(0)#.to(inputs.device)
        
        li_matrix = inputs > average_points_left # 1 if x in inputs is greater that average point, 0 otherwise. shape [__,15]
        ri_matrix = inputs <= average_points_right # 1 if x in inputs is smaller or equal that average point, 0 otherwise. shape [__,15]
        
        li_matrix = li_matrix.to(inputs.device)
        ri_matrix = ri_matrix.to(inputs.device)


        one_hot_inputs = torch.logical_and(li_matrix, ri_matrix).to(inputs.device) # tensr that represents onehot encoding of inouts tensor (1 if in the interval, 0 otherwise)
        
        
        one_hot_inputs_left = torch.sum(distance_points_left*one_hot_inputs, dim = 1).unsqueeze(1).to(inputs.device) #[1200,1]
        
        
        one_hot_inputs_right = torch.sum(distance_points_right*one_hot_inputs, dim = 1).unsqueeze(1).to(inputs.device)  #[1200,1]
        

        self.v0 = one_hot_inputs_left 
        self.v1 = one_hot_inputs_right

        v0 = inputs  - one_hot_inputs_left #  [12000,15]
        #v0 = torch.sum(v0,dim = 1)
        v0 = v0.reshape(a,1,b)     
        v1 = inputs + one_hot_inputs_right #  [12000,15]
        #v1 = torch.sum(v1,dim = 1)
        v1 = v1.reshape(a,1,b)
        #print("shape of v1: ", v1.shape)
        return v0 , v1
        
        


  
            
    
    
        

    
    
         

    @torch.jit.unused
    def _likelihood(self, inputs: Tensor, sp: bool = False):   

  
        #average_points = self.sos.calculate_average_points() # punti-medi per ogni livello di quantizzazione 
        #distance_points = self.sos.calculate_distance_points() # distanza tra i punti di quantizzazione

        average_points = self.sos.average_points
        distance_points = self.sos.distance_points


        v0,v1 = self.define_v0_and_v1(inputs, average_points, distance_points)
        #v0 = float(0.5)
        #v1 = float(0.5)


        lower = self._logits_cumulative(v0 , stop_gradient= sp)
        upper = self._logits_cumulative(v1 , stop_gradient=sp)
            
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
        torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        
        return likelihood

    





    def forward(self, x,  training = True):

        perm, inv_perm = self.define_permutation(x)



        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1).to(x.device)

        outputs = self.quantize(values,"training" if training else "dequantize")
       

        sp = not training
        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs, sp = sp)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
        
        outputs =outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs,  likelihood





    def order_pars(self):
        self.sos.w = torch.nn.Parameter(torch.sort(self.sos.w)[0])
        self.sos.b = torch.nn.Parameter(torch.sort(self.sos.b)[0])
    
    def compress_torcach(self, x, perms):
        perm = perms[0]
        inv_perm = perms[1]
        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1) #[192,1,-----]
        x = self.quantize(values,"symbols") #[1,192,32,48]
        x = x.reshape(shape)
        x = x.permute(*inv_perm).contiguous()

        return super().compress_torcach(x) 




    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        return super().decompress(strings, indexes)




    def decompress_torcach(self, byte_stream, output_cdf):
        outputs = torchac.decode_float_cdf(output_cdf, byte_stream) 
        outputs = self.dequantize(outputs)
        return outputs





