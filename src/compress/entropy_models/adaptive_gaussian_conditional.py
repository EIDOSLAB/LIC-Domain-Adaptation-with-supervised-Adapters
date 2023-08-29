
import torch.nn as nn 
import torch 
import numpy as np
import copy
from typing import Any, Callable, List, Optional, Tuple, Union 

import scipy.stats
from torch import Tensor
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ops import LowerBound
import torch.nn.functional as F
from compress.quantization.activation import AdaptiveQuantization
import torchac

from compress.entropy_models.coder import _EntropyCoder, default_entropy_coder, pmf_to_quantized_cdf, _forward

class AdaptedEntropyModel(nn.Module):
    r"""Entropy model base class.
    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
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

    # See: https://github.com/python/mypy/issues/8795
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



    def quantize(self, inputs, mode,  means = None, permutation= False):



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

        outputs = inputs.clone()
        #if means is not None:
        #    outputs -= means

        if mode == "training":
            outputs = self.sos(outputs, mode)  

            
            #if means is not None:
            #    outputs += means
            if permutation is True:
                outputs =outputs.reshape(shape)
                outputs = outputs.permute(*perms[1]).contiguous()
            

            return outputs 
        
        # outputs = inputs.clone()
        #if means is not None:
        #    outputs -= means


        if means is not None and mode == "symbols":
            outputs -= means     


        outputs = self.sos( outputs, mode) 
        


        if mode == "dequantize":
            #if means is not None:
            #    outputs += means

            if permutation is True:
                outputs =outputs.reshape(shape)
                outputs = outputs.permute(*perms[1]).contiguous()
            return outputs #outputs

        if permutation is True:
            outputs =outputs.reshape(shape)
            outputs = outputs.permute(*perms[1]).contiguous()


        assert mode == "symbols", mode
        shape_out = outputs.shape
        outputs = outputs.ravel()
        map_float_to_int = self.sos.map_sos_cdf 
        
        for i in range(outputs.shape[0]):
            outputs[i] =  int(self.transform_map(outputs[i], map_float_to_int))

        outputs = outputs.reshape(shape_out)    
        return outputs



    """
    def map_to_level(self, inputs, maps, dequantize = False):
        shape_out = inputs.shape
        outputs = inputs.ravel()
        for i in range(outputs.shape[0]):
            if dequantize is False:
                outputs[i] =  self.transform_map(outputs[i], maps)
            else: 
                outputs[i] =   torch.from_numpy(np.asarray(self.transform_map(outputs[i], maps))).to(outputs.device)
        outputs = outputs.reshape(shape_out)
        outputs = outputs.int()   
        return outputs 
    """    
        

     
    def dequantize(self, inputs, means = None, dtype = torch.float):
        """
        we have to 
        1 -map again the integer values to the real values for each channel
        2 - ad the means  
        """
        inputs = inputs.to(dtype)
        outputs =  copy.deepcopy(inputs.detach())
        #print("valori unici di input 1: ",torch.unique(inputs))
        #print("la mappa è   ", self.sos.map_cdf_sos)
        map_int_to_float = self.sos.map_cdf_sos
        shape_inp = outputs.shape
        outputs = outputs.ravel()
        for i in range(outputs.shape[0]):
            c = torch.tensor(map_int_to_float[outputs[i].item()],dtype=torch.float32)
            outputs[i] = c.item()
        outputs = outputs.reshape(shape_inp)

       

        if means is not None:
            outputs = outputs.type_as(means)
            outputs += means
        outputs = outputs.type(dtype)
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

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    

    def retrieve_cdf_from_indexes(self, shapes,indexes): 
        output_cdf = torch.zeros(shapes)
        output_cdf = output_cdf[:,None] + torch.zeros(self.cdf.shape[1])
        output_cdf = output_cdf.to("cpu")
        for i in range(shapes):
            output_cdf[i,:] = self.cdf[indexes[i].item(),:]  
        return output_cdf 

    def compress(self, inputs, indexes):


        symbols = inputs #[1,128,32,48]
        shape_symbols = symbols.shape


        symbols = symbols.ravel().to(torch.int16)
        indexes = indexes.ravel().to(torch.int16)

        
        symbols = symbols.to("cpu")  

        output_cdf = torch.zeros_like(symbols)
        output_cdf = output_cdf[:,None] + torch.zeros(self.cdf.shape[1])
        output_cdf = output_cdf.to("cpu")
        for i in range(symbols.shape[0]):
            output_cdf[i,:] = self.cdf[indexes[i].item(),:]
        byte_stream = torchac.encode_float_cdf(output_cdf, symbols, check_input_bounds=True)

        #c = torchac.decode_float_cdf(output_cdf, byte_stream)
        #if torchac.decode_float_cdf(output_cdf, byte_stream).equal(symbols) is False:
        #    raise ValueError("L'output Gaussiano codificato è diverso, qualcosa non va!")
        #else:
        #    print("l'immagine è ok!")
        return byte_stream, output_cdf  , shape_symbols





    def compress_torcach(self, inputs, indexes):
        symbols = inputs #[1,128,32,48]
        shape_symbols = symbols.shape
        symbols = symbols.ravel().to(torch.int16)
        indexes = indexes.ravel().to(torch.int16)     
        symbols = symbols.to("cpu")  
        output_cdf = torch.zeros_like(symbols)
        output_cdf = output_cdf[:,None] + torch.zeros(self.cdf.shape[1])
        output_cdf = output_cdf.to("cpu")
        for i in range(symbols.shape[0]):
            output_cdf[i,:] = self.cdf[indexes[i].item(),:]
        byte_stream = torchac.encode_float_cdf(output_cdf, symbols, check_input_bounds=True)
        return byte_stream, output_cdf, shape_symbols  



    
  


class AdaptedGaussianConditional(AdaptedEntropyModel):


    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        channels: int = 128,      
        extrema: int = 10,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        trainable = True,
        device = torch.device("cuda"),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

        self.channels = int(channels)
        self.M = int(channels)

        self.sos = AdaptiveQuantization(extrema = extrema, trainable = trainable, device = device)


          
    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)


    def update_scale_table(self, scale_table, force = True):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        #self.update()
        return True









    
    def update(self, force = True , device = torch.device("cuda")):

        self.sos.update_state(device)
        max_length = self.sos.cum_w.shape[0]
        
        pmf_l = self.scale_table.shape[0]  # questi sono il numero  deviazioni standard considerate
        pmf_length = torch.zeros(pmf_l).int().to(device) 
        pmf_length += max_length
        pmf_length = pmf_length.unsqueeze(1) 

        self.sos.define_channels_map()


        average_points = self.sos.average_points # punti-medi per ogni livello di quantizzazione 
        distance_points = self.sos.distance_points

        samples = self.sos.cum_w
        samples = samples.repeat(self.scale_table.shape[0],1)
        samples = samples.to(device)




        low,up = self.define_v0_and_v1(samples, average_points, distance_points)
        low = low.to(samples.device)
        up = up.to(samples.device)


        samples_scale = self.scale_table.unsqueeze(1)  #[64,1]
        samples = samples.float()
        #samples = torch.abs(samples) # da correggerre 
        samples_scale = samples_scale.float()
        

        # adapt to non simmetric quantization steps 
        upper_pos = self._standardized_cumulative((low - samples) / samples_scale)*(samples >= 0)
        upper_neg = self._standardized_cumulative((samples + up) / samples_scale)*(samples < 0)
        lower_pos = self._standardized_cumulative((-up  - samples) / samples_scale)*(samples >= 0)
        lower_neg = self._standardized_cumulative(( samples - low) / samples_scale)*(samples < 0)
            
        upper = upper_pos + upper_neg
        lower = lower_pos + lower_neg
            
        pmf = upper - lower

        # proviamo a fare andare anche il loro entropy coder, se fosse così sarebbe bellissimo 
        tail_mass = 2 * lower[:, :1] # lower questo però è molto molto alto per valori a sigma molto alti
        #tail_mass = torch.zeros(64,1).to(samples.device)# 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset =  torch.zeros(samples[:,0].shape).to(samples.device) # valori minimi possibili!
        #self._offset -= self.sos.cum_w[0].item()
   
   
        self._offset = self._offset.int()
        self._cdf_length = pmf_length + 2

        self._cdf_length = self._cdf_length.squeeze(1)

        ###################################################################################
        self.pmf = pmf
        self.cdf =  self.pmf_to_cdf()





    def pmf_to_cdf(self):
        cdf = self.pmf.cumsum(dim=-1)
        spatial_dimensions = self.pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=self.pmf.dtype, device=self.pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)
        return cdf_with_0
         

    
    def define_v0_and_v1(self, inputs, average_points, distance_points): 


        inputs_shape = inputs.shape
        inputs = inputs.reshape(-1) #.to(inputs.device) # perform reshaping 
        inputs = inputs.unsqueeze(1)#.to(inputs.device) # add a dimension
       
        #average_points = average_points.to(inputs.device)
        #distance_points = distance_points.to(inputs.device)
       
        
        average_points_left = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) - 1000 # 1000 è messo a caso al momento 
        average_points_left[1:] = average_points
        average_points_left = average_points_left.unsqueeze(0).to(inputs.device)
        

        average_points_right = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) + 1000 # 1000 è messo a caso al momento 
        average_points_right[:-1] = average_points
        average_points_right = average_points_right.unsqueeze(0).to(inputs.device)       
               
               
        distance_points_left = torch.cat((torch.tensor([0]).to(inputs.device),distance_points),dim = -1).to(inputs.device)
        distance_points_left = distance_points_left.unsqueeze(0).to(inputs.device)
        
        distance_points_right = torch.cat((distance_points, torch.tensor([0]).to(inputs.device)),dim = -1).to(inputs.device)
        distance_points_right = distance_points_right.unsqueeze(0).to(inputs.device)
        
        li_matrix = inputs > average_points_left # 1 if x in inputs is greater that average point, 0 otherwise. shape [__,15]
        ri_matrix = inputs <= average_points_right # 1 if x in inputs is smaller or equal that average point, 0 otherwise. shape [__,15]
        
        #li_matrix = li_matrix#.to(inputs.device)
        #ri_matrix = ri_matrix#.to(inputs.device)

        one_hot_inputs = torch.logical_and(li_matrix, ri_matrix).to(inputs.device) # tensr that represents onehot encoding of inouts tensor (1 if in the interval, 0 otherwise)
              
        one_hot_inputs_left = torch.sum(distance_points_left*one_hot_inputs, dim = 1).unsqueeze(1).to(inputs.device) #[1200,1]
        one_hot_inputs_right = torch.sum(distance_points_right*one_hot_inputs, dim = 1).unsqueeze(1).to(inputs.device) #[1200,1]
        
        
        v0 = one_hot_inputs_left.reshape(inputs_shape)#.to(inputs.device) #  in ogni punto c'è la distanza con il livello a sinistra       
        v1 = one_hot_inputs_right.reshape(inputs_shape)#.to(inputs.device) # in ogni punto c'è la distanza con il livello di destra

        return v0 , v1


    #add something
    def _likelihood(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None):


        #average_points = self.sos.average_points#.to(inputs.device)
        #distance_points = self.sos.distance_points#.to(inputs.device)

        if means is not None:
            values = inputs - means
        else:
            values = inputs
        
        #values = torch.abs(values)
        low,up = self.sos.define_v0_and_v1(values)
        
        #low = low.to(inputs.device)
        #up = up.to(inputs.device)


        #values = values.to(inputs.device)
        #values = torch.abs(values).to(inputs.device)


        scales = self.lower_bound_scale(scales)

        upper_pos = self._standardized_cumulative((low - values) / scales)*(values >= 0)
        upper_neg = self._standardized_cumulative((values + up) / scales)*(values < 0)
        lower_pos = self._standardized_cumulative((-up  - values) / scales)*(values >= 0)
        lower_neg = self._standardized_cumulative(( values - low) / scales)*(values < 0)
        

        upper = upper_pos  + upper_neg
        lower = lower_pos + lower_neg
        
        #lower = lower_pos + lower_neg
        likelihood = upper - lower

        #upper =self._standardized_cumulative((low - values) / scales)
        #lower = self._standardized_cumulative((-up - values) / scales)

        #likelihood = upper - lower
        return likelihood





    def forward(self, x ,scales , training = True, means = None, adapter = None):


        self.sos.update_state()
        perm, inv_perm = self.define_permutation(x)
        perms = [perm, inv_perm]


        #if means is not None:
        #    values = values #x - means 
        #else: 
        #    values = x


        values = x

        values =  values.permute(*perms[0]).contiguous() # flatten y and call it values
        shape = values.size() 
        values = values.reshape(1, 1, -1) # reshape values
        if means is not None:   
            means = means.permute(*perms[0]).contiguous()
            means = means.reshape(1, 1, -1)#.to(x.device)     




        y_hat = self.quantize(values, "training" if training else "dequantize", means = means)
        #y_hat = self.quantize(x, "training" if training else "dequantize", means = means, perms = True)
        
        y_hat = y_hat.reshape(shape)
        y_hat = y_hat.permute(*perms[1]).contiguous()


        #y_hat2 = copy.deepcopy(y_hat.detach())
        #y_hat = adapter(y_hat) + y_hat  # noi vogliamo che questo approcci a quello del modello migliore!


        #equal = torch.allclose(y_hat2, y_hat)
        #print("--> ",equal)

        if means is not None: 
            y_hat = y_hat #+ means 



        if means is not None:
            means = means.reshape(shape)
            means = means.permute(*perms[1]).contiguous()




        values = values.reshape(shape)
        values = values.permute(*perms[1]).contiguous()


        if means is not None: 
            values = values #+ means

        likelihood = self._likelihood(values, scales, means = means)#.to(x.device)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)  
        return y_hat, likelihood 


    def build_indexes(self, scales: Tensor):
        """
        Questa funzione associa ad ogni elemento output scala l'indice corrispondende alla deviazione standard 
        one-to-one mapping tra scala e inyyydexe
        Non è ottimale, perché si associa la scala subito più grande da una lista fissata
        1- la lista fissata mi sembra troppo estesa (serve?)
        """
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()



        #print("--------------------------------")
        #print("il massimo indice relativo a questa immagine è ",torch.max(indexes),"        ", self.scale_table[torch.max(indexes).item()])
        #print("check se nella lista compaiono valori più grandi del punto medio di scale tables")
        #if indexes.ravel() >= int(len(self.scale_table) - 1)/2:
        #    print("le scale in questo caso servono  ")
        #else:
        #    print("bastano scale più piccole")
        
        return indexes



    def permutation_function(self,x):
        perm = np.arange(len(x.shape))
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        return perm, inv_perm







    def compress(self, x, indexes, perms = None, means = None ):

        if perms is not None:
            values =  x.permute(*perms[0]).contiguous() # flatten y and call it values
            shape = values.size() 
            values = values.reshape(1, 1, -1) # reshape values
            if means is not None:
                means =  means.permute(*perms[0]).contiguous() # flatten y and call it values
                means = means.reshape(1, 1, -1) # reshape values
        else:
            values = x
        #print("lo shape di values prima è: ",values.shape)
        x = self.quantize(values, "symbols", means = means)  
        #print("lo shape di x prima è: ",x.shape)

        if perms is not None:
            print("mannaggia a satana io non devo entrare qua!!!")
            x = x.reshape(shape)
            x = x.permute(*perms[1]).contiguous()

        #print("lo shape di x dopo il primo quantiza è: ",x.shape,"    ",indexes.shape)

        return super().compress(x, indexes) 



    def compress_torcach(self, x, indexes, perms, means = None ):
        """
        if perms is not None:
            values =  x.permute(*perms[0]).contiguous() # flatten y and call it values
            shape = values.size() 
            values = values.reshape(1, 1, -1) # reshape values
            if means is not None:
                means =  means.permute(*perms[0]).contiguous() # flatten y and call it values
                means = means.reshape(1, 1, -1) # reshape values
        else:
        """
        values = x
        x = self.quantize(values, "symbols", means = means, permutation = True)  

        return super().compress_torcach(x, indexes) 

    def decompress_torcach(self, byte_stream, output_cdf, shapes = None, means = None):
        outputs =   torchac.decode_float_cdf(output_cdf, byte_stream)
        if shapes is not None:
            print("print inputs shape: ",shapes)
            outputs = outputs.reshape(shapes)
            means = means.reshape(shapes)
        outputs = self.dequantize(outputs, means = means)
        return outputs