import torch 
import torch.nn as nn 
import numpy as np
from compress.layers import conv3x3, subpel_conv3x3, deconv, conv
import torch.nn.functional as F
import torch.nn.init as init


class ZeroLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


@torch.no_grad()
def init_adapter_layer(adapter_layer: nn.Module):
    if isinstance(adapter_layer, nn.Conv2d):
        # adapter_layer.weight.fill_(0.0)
        adapter_layer.weight.normal_(0.0, 0.02)

        if adapter_layer.bias is not None:
            adapter_layer.bias.fill_(0.0)


class Adapter(nn.Module):
    def __init__(self, in_ch = 320, 
                        out_ch = 320, 
                        dim_adapter = 0,
                        stride = 1,
                        kernel_size = 1,
                        groups = 1, 
                        bias = True,
                        standard_deviation = 0.00,
                        mean = 0.0, 
                        padding = 0):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim_adapter = dim_adapter
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = groups 
        self.bias = bias
        self.standard_deviation = standard_deviation
        self.mean = mean 
        self.padding = padding
        



        self.AdapterModule = self.define_adapter()




    def reinitialize_adapter(self, mean, std):
        nn.init.normal_(self.AdapterModule.weight, mean=mean, std=std)


    def initialization(self,m): 
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)       


    def define_adapter(self):
        if self.dim_adapter == 0:
            if self.stride == 1:
                return ZeroLayer()

            elif self.stride == -1:
                return nn.Sequential(
                    nn.Conv2d(
                        self.in_ch,
                        self.out_ch * 4,
                        kernel_size=1,
                        stride=1,
                        bias=self.bias,
                        groups=self.groups,
                    ),
                    nn.PixelShuffle(2),
                    # above operation is only used for computing the shape.
                    ZeroLayer(),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(
                        self.in_ch,
                        self.out_ch,
                        kernel_size=1,
                        stride= 1, #stride,
                        bias=self.bias,
                        groups=self.groups,
                        padding = self.padding
                    ),
                    # above operation is only used for computing the shape.
                    ZeroLayer(),
                )

        elif self.dim_adapter < 0:
            if self.stride == -1:
                # this implementation of subpixel conv. is done by ours.
                # original impl. by Cheng uses 3x3 conv.
                return nn.Sequential(
                    nn.Conv2d(
                        self.in_ch,
                        self.out_ch * 4,
                        kernel_size=self.kernel_size,
                        stride=1,
                        bias=self.bias,
                        groups=self.groups,
                        padding = self.padding
                    ),
                    nn.PixelShuffle(2),
                )
            else:
                return nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=1, bias=self.bias, groups=self.groups)

        else:
            if self.stride == -1:
                # this implementation of subpixel conv. is done by ours.
                # original impl. by Cheng uses 3x3 conv. ddd
                return nn.Sequential(
                    nn.Conv2d(
                        self.in_ch,
                        self.dim_adapter * 4,
                        kernel_size=self.kernel_size,
                        bias=self.bias,
                        stride=1,
                        groups=self.groups,

                    ),
                    nn.PixelShuffle(2),
                    nn.Conv2d(self.dim_adapter, self.out_ch, kernel_size=self.kernel_size, bias=self.bias, groups=self.groups, stride = 2),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(
                        self.in_ch,
                        self.dim_adapter,
                        kernel_size=self.kernel_size,
                        bias=self.bias,
                        stride=self.stride,
                        groups=self.groups,
                    ),
                    nn.Conv2d(self.dim_adapter, self.out_ch, kernel_size=1, bias=self.bias, groups=self.groups),
                )


    def forward(self,x):
        return  self.AdapterModule(x)
