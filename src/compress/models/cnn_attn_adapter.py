from .cnn import WACNN
import torch.nn as nn
from compressai.layers import GDN
from compressai.models.utils import  deconv, update_registered_buffers, conv
from compress.layers.layers import  Win_noShift_Attention_Adapter, Win_noShift_Attention
from compressai.models import CompressionModel


class WACNNAttentionAdapter(WACNN):
    """
    In questo modello metto gli adapter sia a livello di encoder sia decoder, in maniera speculare 
    """
    def __init__(
        self,
        N=192,
        M=320,
        dim_adapter_1: int = 0,
        dim_adapter_2: int = 1,
        stride_1: int = 1,
        stride_2: int = 1,
        kernel_size_1: int = 1,
        kernel_size_2:int = 1,
        padding_1: int = 0,
        padding_2: int = 0,
        bias: bool = True,
        std: float = 0.00,
        mean: float = 0.00,
        groups: int = 1,
        position: str = "last",
        **kwargs
    ):
        super().__init__(N, M, **kwargs)


        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention_Adapter(
                dim=M,
                num_heads=8,
                window_size=4,
                shift_size=2,
                dim_adapter=dim_adapter_1,
                kernel_size = kernel_size_1,
                stride = stride_1,
                bias = bias,
                groups=groups,
                position=position,
                padding = padding_1,
                std = std,
                mean = mean,
                name = "g_a"
            ),
        )

        """
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention_Adapter(
                dim=N,
                num_heads=8,
                window_size=8,
                shift_size=4,
                dim_adapter=dim_adapter_2,
                kernel_size = kernel_size_2,
                stride = stride_2,
                bias = bias,
                groups=groups,
                position=position,
                padding = padding_2,
                std = std,
                mean = mean,
                name = "g_a"
            ),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention_Adapter(
                dim=M,
                num_heads=8,
                window_size=4,
                shift_size=2,
                dim_adapter=dim_adapter_2,
                kernel_size = kernel_size_2,
                stride = stride_2,
                bias = bias,
                groups=groups,
                position=position,
                padding = padding_2,
                std = std,
                mean = mean,
                name = "g_a"
            ),
        )
        """



        self.g_s = nn.Sequential(
            Win_noShift_Attention_Adapter(
                dim=M,
                num_heads=8,
                window_size=4,
                shift_size=2,
                dim_adapter=dim_adapter_1,
                kernel_size = kernel_size_1,
                stride = stride_1,
                bias = bias,
                groups=groups,
                position=position,
                padding = padding_1,
                std = std,
                mean = mean,
                name = "g_s"
            ),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention_Adapter(
                dim=N,
                num_heads=8,
                window_size=8,
                shift_size=4,
                dim_adapter=dim_adapter_2,
                kernel_size = kernel_size_2,
                padding = padding_2,
                stride = stride_2,
                bias = bias,
                groups=groups,
                position=position,
                std = std,
                mean = mean,
                name = "g_s"
            ),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
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




    def pars_adapter(self, re_grad = False): 

        #for single_adapter in self.adapter_trasforms:
        for n,p in self.g_s.named_parameters(): 
            if "adapter" in n:
                print("nome del parametro: ",n)
                p.requires_grad = re_grad 

        for n,p in self.g_a.named_parameters():
            if "adapter" in n: 
                print("nome del parametro per g_a: ",n)
                p.requires_grad = re_grad
            



    def pars_decoder(self,re_grad = False, st = [0,1,2,3,4,5,6,7,8]):
        

        for i,lev in enumerate(self.g_s):
            if i in  st:
                for n,p in lev.named_parameters():
                    print("nome del parametro: ",n," ",i)
                    p.requires_grad = re_grad
            else:
                print("pass ",i)
