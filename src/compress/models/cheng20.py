from compress.layers.base import ResidualBlock, ResidualBlockUpsample,  AttentionBlock

from compressai.models.utils import  update_registered_buffers
from compressai.models import Cheng2020Attention
from compressai.models import CompressionModel
from torch import nn
from compress.layers.gate import GateNetwork
from compress.layers.layers import AttentionBlockWithMultipleAdapters, ResidualBlockUpsampleMultipleAdapters, ResidualBlockMultipleAdapters, subpel_conv3x3MultipleAdapters
import torch
import torch.nn.functional as F

class Cheng2020AttnAdapter(Cheng2020Attention):
    def __init__(
        self,
        N=192,
        num_adapter = 3,
        dim_adapter_attn: list = [0,0,0],
        stride_attn: list = [1,1,1],
        kernel_size_attn:list = [1,1,1],
        padding_attn: list = [0,0,0],
        position_attn: list = ["res","res","res"],
        type_adapter_attn: list = ["singular","singular","singular"],
        aggregation: str = "top1",
        bias: bool = True,
        std: float = 0.00,
        mean: float = 0.00,
        groups: int = 1,
        **kwargs
    ):
        super().__init__(N, **kwargs)


        self.gate = GateNetwork(in_dim= N, mid_dim = int(N//3),num_adapter=num_adapter)

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlockWithMultipleAdapters(N,
                                            dim_adapter=dim_adapter_attn,
                                            kernel_size = kernel_size_attn,
                                            padding = padding_attn,
                                            stride = stride_attn,
                                            bias = bias,
                                            std = std,
                                            position = position_attn,
                                            type_adapter = type_adapter_attn,
                                            mean = mean, 
                                            groups = groups,
                                            num_adapter = num_adapter,
                                            aggregation = aggregation
                                            ),                                             
            ResidualBlockMultipleAdapters(N,
                                          N,  
                                          num_adapter = num_adapter,  
                                          aggregation= aggregation),
            ResidualBlockUpsampleMultipleAdapters(N, N, 2,
                                          num_adapter = num_adapter,  
                                          aggregation= aggregation),
            ResidualBlockMultipleAdapters(N,
                                          N,  
                                          num_adapter = num_adapter,  
                                          aggregation= aggregation),
            subpel_conv3x3MultipleAdapters(N, 3, 2),
        )


    def load_state_dict(self, state_dict, strict: bool = True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super(CompressionModel, self).load_state_dict(state_dict, strict=strict)


    
    def forward(self, x,  oracle = None):
        y = self.g_a(x)


        gate_values = self.gate(y) #questi sono i valori su cui fare la softmax 
        if x.shape[0]> 0:
            gate_probs = F.softmax(gate_values, dim = 1)
        else:
            gate_probs = F.softmax(gate_values, dim = 0)


        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)





        for j,module in enumerate(self.g_s):
            if j in (0,1,2,3,4):
                y_hat = module(y_hat)
            elif j in (5,6,7,8):
                y_hat = module(y_hat, gate_probs, oracle)
            else: # caso finale in cui j == self.length_reconstruction_decoder -1
                x_hat = module(y_hat, gate_probs,oracle)



        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "logits": gate_values, 
            "y_hat":y_hat
        }