from .cnn_latent_adapter import WACNNStanh
import torch.nn as nn
from compressai.layers import GDN
from compressai.models.utils import  deconv
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.layers.layers import  Win_noShift_Attention_Adapter
import torch 
from compress.ops import ste_round


class WACNNDecoderAdapter(WACNNStanh):
    def __init__(
        self,
        gaussian_configuration = None,
        N=192,
        M=320,

        dim_adapter_1_attention: int = 1,
        dim_adapter_2_attention: int = 1,
        groups: int = 1,
        position: str = "last",
        stride_1: int = 1,
        stride_2: int = 1,
        **kwargs
    ):
        super().__init__(gaussian_configuration = gaussian_configuration ,N=N, M=M, **kwargs)


        self.g_s = nn.Sequential(
            Win_noShift_Attention_Adapter(dim=M,num_heads=8,window_size=4,shift_size=2,dim_adapter=dim_adapter_1_attention,groups=groups,position=position, stride = stride_1),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention_Adapter(dim=N,num_heads=8,window_size=8,shift_size=4,dim_adapter=dim_adapter_2_attention,groups=groups,position=position,stride = stride_2),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)


    def pars_adapter(self, re_grad = True):
        for n,p in self.g_s.named_parameters():
            if "adapter" in n: 
                print(n)
                p.requires_grad = re_grad

    def forward(self, x, training = True):

        self.gaussian_conditional.sos.update_state(x.device) # update state

        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)


        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)




        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        y_teacher = []
        y_hat_no_mean = []


        

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            #y_hat_slice, y_slice_likelihood = self.gaussian_conditional(y_slice, training = training, scales = scale, means = mu, adapter = self.adapter_trasforms[slice_index])  
            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(y_slice, 
                                                                        training = training, 
                                                                        scales = scale, 
                                                                        means = mu)  

            # y_hat_slice ha la media, togliamola per la parte di loss riservata all'adapter 
            y_hat_slice_no_mean = y_hat_slice - mu
            y_hat_no_mean.append(y_hat_slice_no_mean)



            y_likelihood.append(y_slice_likelihood)

            #y_hat_slice = self.gaussian_conditional.quantize(y_slice,mode = "dequantize",means = mu, permutation = True) # sos(y -mu, -1) + mu
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

            y_teacher.append(ste_round(y_slice - mu))  # non devo riaggiungere la media


        
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_teacher = torch.cat(y_teacher, dim = 1)

        y_hat_no_mean = torch.cat(y_hat_no_mean, dim = 1)



        #if  self.decoding_adapter is not None:
        #    y_hat = self.decoding_adapter(y_hat) + y_hat  #con di_adapter = 0 questo non ha ripercussioni sul risultato (controllare)

       
        x_hat = self.g_s(y_hat)


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y_hat": y_hat,
            "y": y,
            "z":z,
            "y_teacher":y_teacher,
            "y_hat_no_mean": y_hat_no_mean

        }
    



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
            y_q_slice_encoded = self.gaussian_conditional.quantize(y_slice, "symbols", mu, permutation= True)


            
            y_hat_slice = self.gaussian_conditional.dequantize(y_q_slice_encoded, mu)  # inverse_map(y_q_slice) + mu






            symbols_list.extend(y_q_slice_encoded.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)


        symbols_list =  [int(x) for x in symbols_list]



        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "z":z,"y":y }
    



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
            #print("decoded slice ",slice_index)
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


        #print("prima i valori sono---> ",torch.unique(y_hat_slice))
        #y_hat_slice = torch.round(y_hat_slice)
        #print("DOPO i valori sono---> ",torch.unique(y_hat_slice))
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}