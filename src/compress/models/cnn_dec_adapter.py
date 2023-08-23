from .cnn_latent_adapter import WACNNStanh
import torch.nn as nn
from compressai.layers import GDN
from compressai.models.utils import  deconv
from compress.layers import  Win_noShift_Attention_Adapter



class WACNNDecoderAdapter(WACNNStanh):
    def __init__(
        self,
        N=192,
        M=320,
        dim_adapter_1: int = 0,
        dim_adapter_2: int = 1,
        groups: int = 1,
        position: str = "last",
        **kwargs
    ):
        super().__init__(N, M, **kwargs)
        self.g_s = nn.Sequential(
            Win_noShift_Attention_Adapter(
                dim=M,
                num_heads=8,
                window_size=4,
                shift_size=2,
                dim_adapter=dim_adapter_1,
                groups=groups,
                position=position,
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
                groups=groups,
                position=position,
            ),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
