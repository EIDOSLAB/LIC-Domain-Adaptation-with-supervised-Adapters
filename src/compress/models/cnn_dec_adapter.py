from .cnn import WACNN
import torch.nn as nn

from compressai.models.utils import  update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.layers.layers import  Win_noShift_Attention_Adapter, deconv,   adapter_res
from compress.layers.gdn import GDN, GDN_Adapter
import torch 
from compressai.models import CompressionModel


class WACNNDecoderAdapter(WACNN):
    def __init__(
        self,
        N=192,
        M=320,
        dim_adapter_attn_1: int = 0,
        stride_attn_1: int = 1,
        kernel_size_attn_1: int = 1,
        padding_attn_1: int = 0,
        position_attn_1: str = "res_last",
        type_adapter_attn_1: str = "singular",

        dim_adapter_attn_2: int = 1,
        stride_attn_2: int = 1,
        kernel_size_attn_2:int = 1,
        padding_attn_2: int = 0,
        position_attn_2: str = "res_last",
        type_adapter_attn_2: str = "singular",



        dim_adapter_deconv_1: int = 0,
        stride_deconv_1: int = 1,
        kernel_size_deconv_1: int = 1,
        padding_deconv_1: int = 0,
        type_adapter_deconv_1: str = "singular",

        dim_adapter_deconv_2: int = 1,
        stride_deconv_2: int = 1,
        kernel_size_deconv_2:int = 1,
        padding_deconv_2: int = 0,
        type_adapter_deconv_2: str = "singular",



        bias: bool = True,
        std: float = 0.00,
        mean: float = 0.00,
        groups: int = 1,

        **kwargs
    ):
        super().__init__(N, M, **kwargs)

        
        self.g_s = nn.Sequential(
            Win_noShift_Attention_Adapter( dim=M, num_heads=8,window_size=4,shift_size=2, # for the attention (no change)
                                          dim_adapter=dim_adapter_attn_1,
                                          kernel_size = kernel_size_attn_1,
                                          stride = stride_attn_1,
                                          bias = bias,
                                          padding = padding_attn_1,
                                          std = std, 
                                          position = position_attn_1,
                                          type_adapter = type_adapter_attn_1,
                                          mean = mean, 
                                          groups = groups),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention_Adapter(dim=N,num_heads=8,window_size=8,shift_size=4, # for the attention (no change)
                                          dim_adapter=dim_adapter_attn_2,
                                          kernel_size = kernel_size_attn_2,
                                          padding = padding_attn_2,
                                          stride = stride_attn_2,
                                          bias = bias,
                                          std = std,
                                          position = position_attn_2,
                                          type_adapter = type_adapter_attn_2,
                                          mean = mean, 
                                          groups = groups),
            deconv(N, N, kernel_size=5, stride=2),
            adapter_res(N,
                        dim_adapter = dim_adapter_deconv_1, 
                        padding =padding_deconv_1, 
                        stride= stride_deconv_1, 
                        std = 0.0,
                        mean = mean ,
                        kernel_size =kernel_size_deconv_1,
                        type_adapter = type_adapter_deconv_1,
                        name = "deconv_adapt_2",
                        res = True), 
            GDN(N, inverse=True ), 
            deconv(N, 3, kernel_size=5, stride=2),
            adapter_res(3,
                        dim_adapter = dim_adapter_deconv_2, 
                        padding =padding_deconv_2, 
                        stride= stride_deconv_2, 
                        std = 0.0,
                        mean = mean ,
                        kernel_size =kernel_size_deconv_2,
                        type_adapter = type_adapter_deconv_2,
                        name = "deconv_adapt_2",
                        res = True) 

       )
        


        """


                    adapter_res(3,
                        dim_adapter = dim_adapter_deconv_2, 
                        padding =padding_deconv_2, 
                        stride= stride_deconv_2, 
                        std = 0.0,
                        mean = mean ,
                        kernel_size =kernel_size_deconv_2,
                        type_adapter = type_adapter_deconv_2,
                        name = "deconv_adapt_2",
                        res = True) 

        self.g_s = nn.Sequential(
            Win_noShift_Attention_Adapter(dim=M,num_heads=8,window_size=4,shift_size=2,dim_adapter=dim_adapter_1,kernel_size = kernel_size_1,stride = stride_1,bias = bias,padding = padding_1,std = std),
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
                mean = mean
            ),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )    

        """


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
                    print("sto sbloccando gli adapter: ",n)
                    p.requires_grad = re_grad 
            



    def pars_decoder(self,re_grad = False, st = [0,1,2,3,4,5,6,7,8]):
        

        for i,lev in enumerate(self.g_s):
            if i in  st:
                for n,p in lev.named_parameters():
                    print("nome del parametro: ",n," ",i)
                    p.requires_grad = re_grad
            else:
                print("pass ",i)




    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        
        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])




        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


                
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    


    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)



            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        
            
        y_hat = torch.cat(y_hat_slices, dim=1)


        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat, "y_hat":y_hat}