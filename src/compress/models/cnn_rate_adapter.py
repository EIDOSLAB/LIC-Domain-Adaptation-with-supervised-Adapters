import math
import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.entropy_models import EntropyBottleneck
from compress.ops import ste_round
from .cnn import WACNN
from compress.entropy_models import  AdaptedEntropyBottleneck, AdaptedGaussianConditional
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




class WACNNRateAdapter(WACNN): 
    def __init__(self, gaussian_configuration,N=192, M=320, factorized_configuration = None,  dim_adapter = 0,stride = 1, **kwargs):
        super().__init__(N = N , M = M,**kwargs)
        
        self.factorized_configuration = factorized_configuration
        self.gaussian_configuration = gaussian_configuration
        self.dim_adapter = dim_adapter 
        self.stride = stride 




        if self.factorized_configuration is None:
            self.entropy_bottleneck = EntropyBottleneck(N)
        else:
            self.entropy_bottleneck = AdaptedEntropyBottleneck(N,      
                                                extrema = self.factorized_configuration["extrema"],
                                                trainable = self.factorized_configuration["trainable"],
                                                device = torch.device("cuda") 
                                                )   

        self.gaussian_conditional = AdaptedGaussianConditional(None,
                                                            channels = N,
                                                            extrema = self.gaussian_configuration["extrema"], 
                                                            trainable =  self.gaussian_configuration["trainable"],
                                                            device = torch.device("cuda")
                                                            )
        

        
        
        #self.adapter_trasforms  = nn.ModuleList( Adapter(32, 32 , dim_adapter= self.dim_adapter) for i in range(10) )
        #self.adapter_scale = Adapter(320, 320 , dim_adapter= self.dim_adapter, stride = self.stride )

        self.adapter_scale_trasforms_loop  = nn.ModuleList( Adapter(32, 32 , dim_adapter= 0, stride= 1) for i in range(10) ) # all'inizio non fanno nulla 
        self.adapter_mean_trasforms_loop  = nn.ModuleList( Adapter(32, 32 , dim_adapter= 0, stride= 1) for i in range(10) ) # all'inizio non fanno nulla
        
        
        self.pars_adapter()
        




    def count_ad_pars(self):
        cc = 0
        for p in self.adapter.parameters(): 
            cc +=1
        return cc

    def modify_adapter(self, args, device):
        print("io qua sto entrando con questo ",args.dim_adapter)
        self.dim_adapter = args.dim_adapter 

        self.adapter_scale_trasforms_loop  = nn.ModuleList( Adapter(32, 32 , 
                                                                    dim_adapter= self.dim_adapter, 
                                                                    standard_deviation= args.std,
                                                                    type_adapter= args.type_adapter,
                                                                    mean= args.mean, 
                                                                    bias = args.bias, 
                                                                    kernel_size = args.kernel_size, 
                                                                    stride = args.adapter_stride, 
                                                                    padding = args.padding )
                                                                    for i in range(10) )
        



        self.adapter_mean_trasforms_loop  = nn.ModuleList( Adapter(32, 32 , 
                                                                    dim_adapter= self.dim_adapter, 
                                                                    standard_deviation= args.std,
                                                                    type_adapter= args.type_adapter,
                                                                    mean= args.mean, 
                                                                    bias = args.bias, 
                                                                    kernel_size = args.kernel_size, 
                                                                    stride = args.adapter_stride, 
                                                                    padding = args.padding )
                                                                    for i in range(10) )


        self.adapter_scale_trasforms_loop.to(device)
        self.adapter_scale.to(device)
        self.pars_adapter(re_grad = True)








    def parse_hyperprior(self, unfreeze_hsa = False, unfreeze_hsa_loop = False):

        for n,p in self.h_scale_s.named_parameters():
            p.requires_grad = unfreeze_hsa

        for single_module in self.cc_scale_transforms:
            for n,p in single_module.named_parameters():
                p.requires_grad = unfreeze_hsa_loop



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
        for single_adapter in self.adapter_mean_trasforms_loop:
            for p in single_adapter.parameters():
                p.requires_grad = re_grad 



        for single_adapter in self.adapter_scale_trasforms_loop:
            for p in single_adapter.parameters():
                p.requires_grad = re_grad 


    def print_information(self):
        print(" h_means_a: ",sum(p.numel() for p in self.h_mean_s.parameters()))
        print(" h_scale_a: ",sum(p.numel() for p in self.h_scale_s.parameters()))
        print("cc_mean_transforms",sum(p.numel() for p in self.cc_mean_transforms.parameters()))
        print("cc_scale_transforms",sum(p.numel() for p in self.cc_scale_transforms.parameters()))
        


                



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



    def forward_adapter(self, x): 




        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)


        z_hat, z_likelihoods = self.entropy_bottleneck(z, training = False)




        #latent_scales = self.h_scale_s(z_hat)
        #latent_means = self.h_mean_s(z_hat)




        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        #latent_scales = self.adapter_scale(latent_scales) + latent_scales



        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []




        for slice_index, y_slice in enumerate(y_slices):

            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            mu = self.adapter_mean_trasforms_loop[slice_index](mu) + mu

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            scale = self.adapter_scale_trasforms_loop[slice_index](scale) + scale

            #y_hat_slice, y_slice_likelihood = self.gaussian_conditional(y_slice, training = False, scales = scale, means = mu) # solo per mostrare la likelihood
            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(y_slice, training = False, scales = scale, means = mu)


            #perm, inv_perm = self.define_permutation(y_slice)
            #perms = [perm,inv_perm]
            #y_hat_slice = self.gaussian_conditional.quantize(y_slice,mode = "dequantize", means = mu, perms = perms)
            
            y_likelihood.append(y_slice_likelihood)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)


        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices, dim=1)
        


        # da mettere qua l'adapter? 
        #y_hat = self.adapter(y_hat) + y_hat  #con di_adapter = 0 questo non ha ripercussioni sul risultato (controllare)

       
        x_hat = self.g_s(y_hat)


        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y_hat": y_hat,
            "y": y,
            "z":z,
            "z_hat": z_hat,
            "x":x

        }





    def forward(self, x, training = True):

        if self.factorized_configuration is not  None:
            self.entropy_bottleneck.sos.update_state(x.device)  # update state        
        
        
        self.gaussian_conditional.sos.update_state(x.device) # update state

        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)

        if self.factorized_configuration is not  None:
            z_hat, z_likelihoods = self.entropy_bottleneck(z, training = training)
        else:
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        # vedere le dimensionalità di latent_scales and means (e anche il numero di parametri)

        #latent_scales = self.adapter_scale(latent_scales) + latent_scales


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


            mu = self.adapter_mean_trasforms_loop[slice_index](mu) + mu

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            scale = self.adapter_scale_trasforms_loop[slice_index](scale) + scale

            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(y_slice, training = training, scales = scale, means = mu) #self.adapter_trasforms[slice_index])  

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
        # da mettere qua l'adapter? 
        #y_hat = self.adapter(y_hat) + y_hat  #con di_adapter = 0 questo non ha ripercussioni sul risultato (controllare)

       
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

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=True)
        self.gaussian_conditional.update()
        self.entropy_bottleneck.update(force = True)
        return updated


    def compress_torcach(self,x): 
        y = self.g_a(x)
        y_shape = y.shape[2:]
        #print("Y SHAPE------> ",y_shape)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_cdfs = []
        y_shapes = []
        y_scales =  []
        y_means = []
        y_strings = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            mu = self.adapter_mean_trasforms_loop[slice_index](mu) + mu

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            scale = self.adapter_scale_trasforms_loop[slice_index](scale) + scale

            index = self.gaussian_conditional.build_indexes(scale)
            perm, inv_perm = self.define_permutation(y_slice)
            strings, cdfs, shape_symbols , y_q_slice = self.gaussian_conditional.compress_torcach(y_slice, index,  perms = [perm, inv_perm], means = mu) # shape is flattenend ( in theory) # shape is flattenend ( in theory)
            #y_q_slice = self.gaussian_conditional.quantize(y_slice, mode = "symbols", means = mu, permutation = True) 
            proper_shape = y_q_slice.shape
            
            y_strings.append(strings) 
            y_cdfs.append(cdfs)
            y_shapes.append(proper_shape)
            y_q_slice = self.gaussian_conditional.dequantize(y_q_slice) 
            y_hat_slice = y_q_slice + mu 
                                                             
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)
        
        return {"strings": [ z_strings, y_strings], 
                "cdfs":   y_cdfs,
                "shape":   z.size()[-2:], 
                "params": {"means": y_means, "scales":y_scales}}









    

    def decompress_torcach(self,data):
        strings = data["strings"] 
        y_cdf = data["cdfs"]
        shape = data["shape"]
        
        z_hat = self.entropy_bottleneck.decompress(strings[0], shape)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)



        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[1]
        #y_cdf = cdf

        y_hat_slices = []

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            mu = self.adapter_mean_trasforms_loop[slice_index](mu) + mu

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            #index = self.gaussian_conditional.build_indexes(scale)

            scale = self.adapter_scale_trasforms_loop[slice_index](scale) + scale


            rv = self.gaussian_conditional.decompress_torcach(y_string[slice_index],y_cdf[slice_index]) # decompress -> dequantize  + mu
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            rv = rv.to(mu.device)
            #print("----> ",rv.device,"  ",mu.device)
            y_hat_slice = rv + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
    





    def __deepcopy__(self, memo):
        # Crea una nuova istanza del modello
        new_model = self.__class__(gaussian_configuration= self.gaussian_configuration, factorized_configuration= None, dim_adapter = self.dim_adapter  )
        # Copia i parametri dei moduli
        new_model = new_model.to("cuda")
        
        
        states_d  = copy.deepcopy(self.state_dict(), memo)
        if "entropy_bottleneck._quantized_cdf" in list(states_d.keys()):
            del states_d["entropy_bottleneck._quantized_cdf"]
            del states_d["entropy_bottleneck._cdf_length"]
            del states_d["entropy_bottleneck._offset"]
        if "gaussian_conditional._quantized_cdf" in list(states_d.keys()):
            del states_d["gaussian_conditional._offset"]
            del states_d["gaussian_conditional._quantized_cdf"]
            del states_d["gaussian_conditional._cdf_length"]
            del states_d["gaussian_conditional.scale_table"]
                                  
        new_model.load_state_dict(states_d)
        return new_model