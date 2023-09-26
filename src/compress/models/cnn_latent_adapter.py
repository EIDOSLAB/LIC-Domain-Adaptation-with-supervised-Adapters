import math
import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.entropy_models import EntropyBottleneck, GaussianConditional
from compress.ops import ste_round
from .cnn import WACNN

#from compress.entropy_models.adaptive_gaussian_conditional import GaussianConditionalSoS
#from compress.entropy_models.adaptive_entropy_models import EntropyBottleneckSoS
import torch.nn.functional as F
from compress.quantization.adapter  import Adapter
import copy
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
import numpy as np

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))




class WACNNStanh(WACNN): 
    def __init__(self,N=192, M=320,  dim_adapter = 0,stride = 1, **kwargs):
        super().__init__(N = N , M = M,**kwargs)
        

        self.dim_adapter = dim_adapter 
        self.stride = stride 
        self.entropy_bottleneck = EntropyBottleneck(N)

        self.gaussian_conditional = GaussianConditional(None)

        
        
        #self.adapter_trasforms_loop  = nn.ModuleList( Adapter(32, 32 , dim_adapter= self.dim_adapter) for i in range(10) )
        self.adapter = Adapter(320, 320 , dim_adapter= 0 , stride = 1 )  # empty adapter by now
        #self.pars_adapter()
        



    def print_information(self):
        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        




    def count_ad_pars(self):
        cc = 0
        for p in self.adapter.parameters(): 
            cc +=1
        return cc

    def modify_adapter(self, args, device):
        self.dim_adapter = args.dim_adapter 
        self.adapter = Adapter(320, 320 , 
                               dim_adapter= self.dim_adapter, 
                               standard_deviation= args.std,
                               type_adapter= args.type_adapter,
                                 mean= args.mean, 
                                 bias = args.bias, 
                                 kernel_size = args.kernel_size, 
                                 stride = args.adapter_stride, 
                                 padding = args.padding )
        #self.adapter_trasforms = nn.ModuleList( Adapter(in_ch = 32, out_ch =32, dim_adapter = args.dim_adapter, mean = args.mean, standard_deviation= args.std, kernel_size = args.kernel_size) for i in range(10))
        #self.adapter_trasforms.to(device)
        self.adapter.to(device)
        self.pars_adapter(re_grad = True)



    def parse_hyperprior(self,  re_grad_hma = False, re_grad_hsa = False):
        #for n,p in self.h_a.named_parameters():
        #    p.requires_grad = re_grad_ha  
        for n,p in self.h_mean_s.named_parameters():
            p.requires_grad = re_grad_hma
        for n,p in self.h_scale_s.named_parameters():
            p.requires_grad = re_grad_hsa



    def pars_decoder(self,re_grad = False, st = -1):
        

        for i,lev in enumerate(self.g_s):
            if i > st:
                for n,p in lev.named_parameters():
                    p.requires_grad = re_grad
            else:
                print("pass ",i)

    def freeze_net(self):
        for n,p in self.named_parameters():
            p.requires_grad = False
        
        for p in self.parameters(): 
            p.requires_grad = False

    def unfreeze_quantizer(self): 
        if self.factorized_configuration is not None:
            for p in self.entropy_bottleneck.sos.parameters(): 
                p.requires_grad = True
        for p in self.gaussian_conditional.sos.parameters(): 
            p.requires_grad = True
    

    def freeze_quantizer(self): 

        if self.factorized_configuration is not None:
            for p in self.entropy_bottleneck.sos.parameters(): 
                p.requires_grad = False
        for p in self.gaussian_conditional.sos.parameters(): 
            p.requires_grad = False

    def pars_adapter(self, re_grad = False): 
        #for single_adapter in self.adapter_trasforms:
        for p in self.adapter.parameters(): 
                p.requires_grad = re_grad  
                



    def unfreeze_lrp(self): 
        for block in self.lrp_transforms:
            for layer in block:
                for param in layer.parameters():
                    param.requires_grad = True

     



    def define_permutation(self, x):
        perm = np.arange(len(x.shape)) 
        perm[0], perm[1] = perm[1], perm[0]
        inv_perm = np.arange(len(x.shape))[np.argsort(perm)] # perm and inv perm
        return perm, inv_perm



    def forward(self, x):   
        

        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)


        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        # vedere le dimensionalità di latent_scales and means (e anche il numero di parametri)


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


            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu 

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_teacher = torch.cat(y_teacher, dim = 1)

        y_hat_no_mean = torch.cat(y_hat_no_mean, dim = 1)
        # da mettere qua l'adapter? 
        y_hat = self.adapter(y_hat) + y_hat  #con di_adapter = 0 questo non ha ripercussioni sul risultato (controllare)

       
        x_hat = self.g_s(y_hat)


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y": y, 
            "y_hat":y_hat
        }


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=True)
        self.gaussian_conditional.update()
        self.entropy_bottleneck.update(force = True)
        return updated










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


        y_hat = torch.cat(y_hat_slices, dim=1)


        y_hat = self.adapter(y_hat) + y_hat
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
    

