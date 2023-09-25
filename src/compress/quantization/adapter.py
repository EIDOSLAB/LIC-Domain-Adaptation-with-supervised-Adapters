import torch 
import torch.nn as nn 
from collections import OrderedDict
from .attn_adapter import Transformer, Attention


def reshape_tensor(x, inv = None):
    if inv is None:
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, -1, C), (B,C,H,W)
    else: 
        b,c,h,w = inv[0], inv[1], inv[2], inv[3]
        return  x.permute(0, 2, 1).reshape(b,c,h,w), inv
        #B, WH, C = x.shape
        #H = int(torch.sqrt(torch.tensor(WH / C)).item())
        #W = WH // (C * H)
        #return x.reshape(B, H, W, C).permute(0, 3, 1, 2)       






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
                        type_adapter = "singular",
                        dim_adapter = 0,
                        stride = 1,
                        kernel_size = 1,
                        groups = 1, 
                        bias = True,
                        standard_deviation = 0.00,
                        mean = 0.0, 
                        padding = 0,
                        name = "",
                        res = False,
                        depth = 1):
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
        self.type_adapter = type_adapter
        self.name = name
        self.res = res
        self.depth = depth
        



        self.AdapterModule = self.define_adapter()




    def reinitialize_adapter(self, mean, std):
        nn.init.normal_(self.AdapterModule.weight, mean=mean, std=std)


    def initialization(self,m): 
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear) :
            print("sono entrato qua per inizializzare")
            nn.init.normal_(m.weight, mean=self.mean, std=self.standard_deviation)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)       


    def define_adapter(self):
        if self.type_adapter == "singular":
            if self.dim_adapter == 0:
                if self.stride == 1:
                    print("initially singular on the definition")
                    return ZeroLayer()

                elif self.stride == -1:
                    print("entro dove dovrei entrare")
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
                    print("entro qua a ridefinire l'adapter")
                    # this implementation of subpixel conv. is done by ours.
                    # original impl. by Cheng uses 3x3 conv.
                    model =  nn.Sequential(
                        nn.Conv2d(
                            self.in_ch,
                            self.in_ch * 4,
                            kernel_size=self.kernel_size,
                            stride=2,
                            bias=self.bias,
                            groups=self.groups,
                            padding = self.padding
                        ),
                        nn.PixelShuffle(2),
                    )
                    model.apply(self.initialization)
                    return model 
                else:
                    print("entro qua a ridefinire l'adapter, semplice covoluzione senza nulla!")
                    model = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kernel_size, stride=self.stride, bias=self.bias, groups=self.groups)
                    model.apply(self.initialization)
                    return model #nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kernel_size, stride=self.stride, bias=self.bias, groups=self.groups)
            
            else:
                if self.stride == -1:
                    # this implementation of subpixel conv. is done by ours.
                    # original impl. by Cheng uses 3x3 conv. ddd
                    model = nn.Sequential(
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
                    model.apply(self.initialization)
                    return model
                else:
                    print("entro qua che è il punto in cui faccio come wacv")


                    params = OrderedDict([
                    (self.name + "_adapter_conv1",nn.Conv2d(self.in_ch,self.dim_adapter,kernel_size=self.kernel_size,bias=self.bias,stride=self.stride,groups=self.groups,padding = self.padding)),
                    ('ReLU0',nn.GELU()),
                    (self.name + "_adapter_conv2",nn.Conv2d(self.dim_adapter, self.out_ch, kernel_size=self.kernel_size, bias=self.bias,stride=self.stride, groups=self.groups, padding = self.padding))])
                    
                    
                    model =  nn.Sequential(params)
                    model.apply(self.initialization)

                    return model
        elif self.type_adapter == "attention":
            model = Attention(dim = self.in_ch)
            model.apply(self.initialization)
            return model 

        elif self.type_adapter == "transformer":
            model = Transformer(dim = self.in_ch, depth = self.depth)
            return model

        elif self.type_adapter == "multiple":
            pass
        else:
            raise ValueError("Per ora non ho implementato altro!!")


    def forward(self,x):
        if self.type_adapter == "singular":
            if self.res is False:
                return  self.AdapterModule(x)
            else:
                return x + self.AdapterModule(x)
        elif self.type_adapter == "attention":
            x, inv = reshape_tensor(x)
            out = self.AdapterModule(x)
            out,_ = reshape_tensor(out,inv = inv)
            if self.res is False:
                return out
            else: 
                return x + out
        else: # transformer
            x, inv = reshape_tensor(x) 
            out = self.AdapterModule(x)
            out,_ = reshape_tensor(out,inv = inv)
            return out


            
