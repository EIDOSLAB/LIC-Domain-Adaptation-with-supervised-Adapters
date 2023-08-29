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
                        bias = False,
                        standard_deviation = 0.00,
                        mean = 0.0):
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
            return ZeroLayer()

        elif self.dim_adapter == 1:
            print("sono entrato qua nella definizione")
            convv =  nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=1, bias=self.bias, groups=self.groups)
            nn.init.normal_(convv.weight, mean=self.mean, std=self.standard_deviation)
            if self.bias:
                nn.init.zeros_(convv.bias)
            

            return convv

        elif self.dim_adapter == 2:

            model = nn.Sequential(
                nn.Conv2d(
                    self.in_ch,
                    self.dim_adapter,
                    kernel_size=1,
                    bias=self.bias,
                    stride=self.stride,
                    groups=self.groups,
                ),
                nn.Conv2d(self.dim_adapter, self.out_ch, kernel_size=1, bias=self.bias, groups=self.groups),
            )

            model.apply(self.initialization())
            return model

        else:

            N = self.in_ch
            self.encoder = nn.Sequential(
                conv(N, N, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(N, N, stride = 2, kernel_size = 5),
                nn.ReLU(inplace=True),
                conv(N, N, stride = 2, kernel_size = 5),
            )

            self.decoder = nn.Sequential(
                deconv(N, N, stride = 2, kernel_size = 5),
                nn.ReLU(inplace=True),
                deconv(N, N, stride = 2, kernel_size = 5),
                nn.ReLU(inplace=True),
                conv(N, N, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
            )

            # Inizializzazione dei pesi con valori casuali dalla distribuzione normale con varianza piccola
            for m in self.encoder.modules():
                print("I entered hettrrrre!!!")
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                    init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
                    if m.bias is not None:
                        init.zeros_(m.bias)
            


            # Inizializzazione dei pesi con valori casuali dalla distribuzione normale con varianza piccola
            for m in self.decoder.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
                    if m.bias is not None:
                        init.zeros_(m.bias)
            return [self.encoder, self.decoder]



    def forward(self,x):


   
        if self.dim_adapter !=2:


            return  self.AdapterModule(x)
        else:
            x = self.encoder(x)
            x = self.decoder(x)
            return x
