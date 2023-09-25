import torch 
import torch.nn as nn


in_ch = 320 
out_ch = 320
dim_adapter = 320*2
print(dim_adapter)
kernel_size =1
stride = 1

c =   nn.Conv2d(in_ch,dim_adapter,kernel_size=1,stride=1,padding = 0)
pix = nn.GELU() # nn.PixelShuffle(2)
fin = nn.Conv2d(dim_adapter, out_ch, kernel_size=1, stride=1, padding =0)
d = torch.randn(16,320,16,16)

print(d.shape)
dd = c(d)
print(dd.shape)
dd = pix(dd)
print(dd.shape)
dd = fin(dd)
print(dd.shape)