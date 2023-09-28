# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch import Tensor
import torch
import torch.nn as nn
from .win_attention import WinBasedAttention, WinBaseAttentionAdapter
from compress.adaptation.adapter import Adapter
#from .wacn_adapter import define_adapter, init_adapter_layer
import torch.nn.functional as F
from compressai.ops.parametrizers import NonNegativeParametrizer
from collections import OrderedDict
from .utils_function import  conv1x1, conv3x3

__all__ = [
    "Win_noShift_Attention",
    "AttentionBlock",
    "SelfAttentionResidualBlock"

]



class ResidualAdapterDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, mean = 0, standard_deviation = 0.00, initialize = "gaussian"):
        super().__init__()
        self.original_model_weights = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1,
            padding=kernel_size // 2,
        )
        

        params = OrderedDict([("adapter_transpose_conv1",nn.ConvTranspose2d(in_channels, out_channels,  kernel_size = kernel_size, stride = stride, output_padding = stride -1, padding = kernel_size // 2))])
                    
                    
        
        

        self.Adapterconv =  nn.Sequential(params)
        self.mean = mean 
        self.standard_deviation = standard_deviation
        
        if initialize == "gaussian":
            self.Adapterconv.apply(self.initialization)
    

    def initialization(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear) or isinstance(m, nn.ConvTranspose2d) :
            print("sono entrato qua per inizialissssszzare split connections E SONO QUA DENTRO CON 0")
            nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)       
    
    def forward(self, x):
        x_adapt = self.Adapterconv(x) 
        x_conv = self.original_model_weights(x)
        return x_conv + x_adapt


class ResidualMultipleAdaptersDeconv(nn.Module):
    def __init__(self, in_channels, 
                    out_channels, 
                    kernel_size=5, 
                    stride=2, 
                    mean = 0, 
                    standard_deviation = 0.00,
                    initialize = "gaussian", 
                    num_adapter = 3, 
                    name = [], 
                    aggregation = "top1", 
                    threshold = 0):
        super().__init__()

        self.original_model_weights = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=stride - 1,
            padding=kernel_size // 2,
        )


        self.mean = mean 
        self.standard_deviation = standard_deviation
        self.aggregation = aggregation
        self.threshold = threshold


        self.adapters = nn.ModuleList([]) 
        for i in range(num_adapter):
            if len(name) == 0:
                name_ad = "adapter_" + str(i)
            else: 
                name_ad = "adapter_" + name[i]


            params = OrderedDict([(name_ad,nn.ConvTranspose2d(in_channels, out_channels,  kernel_size = kernel_size, stride = stride, output_padding = stride -1, padding = kernel_size // 2))])  
            adapter =  nn.Sequential(params)
            if initialize == "gaussian":
                adapter.apply(self.initialization)
            self.adapters.append(adapter)


    def initialization(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear) or isinstance(m, nn.ConvTranspose2d) :
            print("sono entrato qua per inizialissssszzare split connections")
            nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias) 


    def forward(self,x, gate_prob):
        x_conv = self.original_model_weights(x)
        if self.aggregation == "top1": #in questo caso prendiamo solo l'adapter migliore e lo sommiamo
            argmax = torch.argmax(gate_prob).item()
            x_adapt = self.adapters[argmax](x)
            return x_conv + x_adapt  
        
        elif self.aggregation == "weighted":
            gate_prob = torch.where(gate_prob < self.threshold, torch.zeros_like(gate_prob), gate_prob)
            x_adapt = torch.sum(torch.stack([self.adapters[i](x)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1)
            return x_conv + x_adapt 
        else: 
            raise ValueError("Per ora non ho implementato altro!!!!")


    























class Win_noShift_Attention(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        N = dim

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out



            def initialize_weights(self, pretrained_layer):
                for i,l in enumerate(pretrained_layer.conv):
                    if i%2 == 0:
                        self.conv[i].weight = pretrained_layer.conv[i].weight
                        self.conv[i].weight.requires_grad = True
                        self.conv[i].bias = pretrained_layer.conv[i].bias
                        self.conv[i].requires_grad = True 
                    else: 
                        continue

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            WinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size),
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

    def initialize_weights(self, pretrained_layer):
        print("starting initializing weights of win_noshift_attention")

        with torch.no_grad():
            for i,l in enumerate(pretrained_layer.conv_b):

                if i == 0:
  
                    continue
                elif i < len(pretrained_layer.conv_b) - 1:
   
                    self.conv_b[i].initialize_weights(pretrained_layer.conv_b[i])
                else:
                    self.conv_b[i].weight = pretrained_layer.conv_b[i].weight
                    self.conv_b[i].weight.requires_grad = True
                    self.conv_b[i].bias = pretrained_layer.conv_b[i].bias
                    self.conv_b[i].requires_grad = True                    
            for i,l in enumerate(pretrained_layer.conv_a):
                self.conv_a[i].initialize_weights(pretrained_layer.conv_a[i])




class Win_noShift_Attention_Adapter(Win_noShift_Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=8,
        shift_size=0,
        dim_adapter: int = 1,
        groups: int = 1,
        position: str = "res_last",
        stride: int = 1,

        kernel_size: int = 1,
        std: float = 0.01,
        mean: float = 0.00,
        bias: bool = False, 
        padding: int = 0,
        name: str = "",
        type_adapter: str = "singular"
    ):
        """Win_noShift_Attention with adapters.

        Args:
            dim_adapter (int): dimension for the intermediate feature of the adapter.
            groups (int): number of groups for the adapter. Default: 1. if groups=dim, the adapter become channel-wise multiplication.
        """
        super().__init__(dim, num_heads, window_size, shift_size)
        self.position = position
        if self.position in ("res_last","last"):
            self.adapter = Adapter(dim, 
                                   dim, 
                                   dim_adapter=dim_adapter, 
                                   groups=groups, 
                                   stride = stride, 
                                   padding = padding, 
                                   standard_deviation= std,
                                   mean=mean,
                                   kernel_size= kernel_size, 
                                   bias = bias,
                                   name = name, 
                                   type_adapter= type_adapter

                                   )
            #self.adapter.apply(init_adapter_layer)
        elif self.position in {"attn", "attnattn"}:
            self.conv_b[0] = WinBaseAttentionAdapter(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                dim_adapter=dim_adapter,
                groups=groups,
                position=self.position,
            )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)

        if self.position == "res_last" :
            # modify output by adapters
            out = out + self.adapter(out)
        elif self.position == "last":
            out = self.adapter(out)


        out += identity
        return out



class Win_noShift_Attention_Multiple_Adapter(Win_noShift_Attention):

    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=8,
        shift_size=0,
        dim_adapter: int = 1,
        groups: int = 1,
        aggregation: str = "top1",
        stride: int = 1,
        num_adapter: int = 3,
        kernel_size: int = 1,
        std: float = 0.01,
        mean: float = 0.00,
        bias: bool = False, 
        padding: int = 0,
        name: list =[],
        type_adapter: str = "singular",
        position: str = "res_last",
        threshold: int = 0
    ):
        """Win_noShift_Attention with multiple adapters.
        """
        super().__init__(dim, num_heads, window_size, shift_size)
        self.aggregation = aggregation
        self.num_adapter = num_adapter
        self.threshold = threshold
        self.position = position
        self.adapters = nn.ModuleList([]) 
        for i in range(num_adapter):

            if len(name) == 0:
                name_ad = "AttentionAdapter_" + str(i)
            else: 
                name_ad = "AttentionAdapter_" + name[i]


            params = OrderedDict([
                    (name_ad,Adapter(dim,
                                    dim, 
                                    dim_adapter=dim_adapter, 
                                    groups=groups, 
                                    stride = stride, 
                                    padding = padding, 
                                    standard_deviation= std,
                                    mean=mean,
                                    kernel_size= kernel_size, 
                                    bias = bias,
                                    name = name, 
                                    type_adapter= type_adapter))])
                                    
            adapter =  nn.Sequential(params)
            self.adapters.append(adapter)



    def aggregate_adapters(self, out, gate_prob):
        #gate_prob = nn.Softmax(gate_output) # estraggo le probabilità delle singole classi
        if self.aggregation == "top1":
            argmax = torch.argmax(gate_prob).item()
            if self.position == "res":
                return out  + self.adapters[argmax](out) # applico solo un adapter! 
            else:
                return self.adapters[argmax](out)
            
        elif self.aggeration == "weighted":
            gate_prob = torch.where(gate_prob < self.threshold, torch.zeros_like(gate_prob), gate_prob)
            if self.position == "res":
                return  out  + torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1) # [BS, num_adapter, h, w]
            else: 
                return torch.sum(torch.stack([self.adapters[i](out)*gate_prob[i] for i in range(self.num_adapter)], dim = 1),dim = 1)
        else: 
            raise ValueError("Per ora non ho implementato altro!!!!")





        




    def forward(self, x, gate_prob):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        return self.aggregate_adapters(out, gate_prob) + identity















class AttentionBlock(nn.Module):
    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit()
                                  # , ResidualUnit()
                                  # , ResidualUnit()
                                    )

        self.conv_b = nn.Sequential(
            ResidualUnit(),
          #  ResidualUnit(),
          #  ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out
    




class GDN(nn.Module):


    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)


        

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out
    




class GDN_Adapter(GDN):


    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        dim_adapter: int = 0,
        stride: int = 1, 
        padding: int = 0, 
        std: float = 0.0,
        mean: float = 0.0,
        name: str = ""
    ):
        super().__init__(in_channels=in_channels, inverse=  inverse, beta_min= beta_min, gamma_init= gamma_init)


        self.adapter_gdn = Adapter(in_channels, in_channels, dim_adapter=dim_adapter, stride= stride, padding=padding, standard_deviation=std,mean=mean, name = name)


    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        norm = norm + self.adapter_gdn(norm)

        out = x * norm

        return out 
    




class SelfAttentionResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionResidualBlock, self).__init__()

        class SelfAttentionModule(nn.Module):
            def __init__(self, in_channels, num_heads=1):
                super(SelfAttentionModule, self).__init__()

                self.attention_multi = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)

            def forward(self, x):
                x = x.permute(2, 0, 1)
                attn_output, _ = self.attention_multi(x, x, x)
                attn_output = attn_output.permute(1, 2, 0)
                return attn_output


        self.attention = SelfAttentionModule(in_channels)

    def forward(self, x):
        out = self.attention(x)  
        # Aggiungi l'input originale al risultato
        #out += residual
        return out
