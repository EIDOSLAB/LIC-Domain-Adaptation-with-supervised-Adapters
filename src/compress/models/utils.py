
import torch
import torch.nn as nn
from compress.zoo import models
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.ops import ste_round


def forward_pass(self: nn.Module, x: torch.Tensor,training: bool = True) -> dict:


        
    self.gaussian_conditional.sos.update_state(x.device) # update state

    y = self.g_a(x)
    y_shape = y.shape[2:]
    z = self.h_a(y)



    _, z_likelihoods = self.entropy_bottleneck(z)

    z_offset = self.entropy_bottleneck._get_medians()
    z_tmp = z - z_offset
    z_hat = ste_round(z_tmp) + z_offset

    latent_scales = self.h_scale_s(z_hat)


    latent_scales = self.h_scale_s(z_hat)
    latent_means = self.h_mean_s(z_hat)


    #print("latent scales shape---> ",latent_scales.shape)

    y_slices = y.chunk(self.num_slices, 1)
    y_hat_slices = []
    y_likelihood = []
    y_hat_no_mean = []
    y_teacher = []

    for slice_index, y_slice in enumerate(y_slices):
        support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[: self.max_support_slices])
        mean_support = torch.cat([latent_means] + support_slices, dim=1)
        mu = self.cc_mean_transforms[slice_index](mean_support)
        mu = mu[:, :, : y_shape[0], : y_shape[1]]
        scale_support = torch.cat([latent_scales] + support_slices, dim=1)
        scale = self.cc_scale_transforms[slice_index](scale_support)
        scale = scale[:, :, : y_shape[0], : y_shape[1]]

        #y_hat_slice, y_slice_likelihood = self.gaussian_conditional( y_slice.clone(), scale, mu)
        
        
        y_hat_slice, y_slice_likelihood = self.gaussian_conditional(y_slice, training =False, 
                                                                    scales = scale, 
                                                                    means = mu, 
                                                                    adapter = self.adapter_trasforms[slice_index])  

        # y_hat_slice ha la media, togliamola per la parte di loss riservata all'adapter 
        y_hat_slice_no_mean = y_hat_slice - mu
        y_hat_no_mean.append(y_hat_slice_no_mean)



        y_likelihood.append(y_slice_likelihood)

        
        
        y_likelihood.append(y_slice_likelihood)

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

    x_hat = self.g_s(y_hat)
    x_hat = x_hat.clamp(0, 1)
    return {
        "y_hat": y_hat,
        "z_hat": z_hat,
        "x_hat": x_hat,
        "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        "y_teacher": y_teacher,
        "y_hat_no_mean":y_hat_no_mean
    }









def rename_key(key):
    """Rename state_deeict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]
    if key.startswith('h_s.'):
        return None

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    # if ".downsample." in key:
    #     return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key
def load_pretrained(state_dict):
    """Convert sccctaddte_dict keys."""
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    if None in state_dict:
        state_dict.pop(None)
    return state_dict


def from_state_dict(cls, state_dict):

    net = cls()#cls(192, 320)
    net.load_state_dict(state_dict)
    return net

def get_model(args,device, N = 192, M = 320, factorized_configuration = None, gaussian_configuration = None  ) -> nn.Module:

    if args.model == "base":
        baseline = True
        print("attn block base method siamo in baseline")
        #net = models[args.model](N = N ,M = M)
        net = models[args.model]()
        if args.pret_checkpoint_base is not None: 

            print("entroa qua per la baseline!!!!")
            #net.update(force = True)
            checkpoint = torch.load(args.pret_checkpoint_base, map_location=device)


            del checkpoint["state_dict"]["entropy_bottleneck._offset"]
            del checkpoint["state_dict"]["entropy_bottleneck._quantized_cdf"]
            del checkpoint["state_dict"]["entropy_bottleneck._cdf_length"]
            del checkpoint["state_dict"]["gaussian_conditional._offset"]
            del checkpoint["state_dict"]["gaussian_conditional._quantized_cdf"]
            del checkpoint["state_dict"]["gaussian_conditional._cdf_length"]
            del checkpoint["state_dict"]["gaussian_conditional.scale_table"]
            
            

                
            state_dict = load_pretrained(torch.load(args.pret_checkpoint_base, map_location=device)['state_dict'])
            net = from_state_dict(models[args.model], state_dict)

            net.update()
            net.to(device) 


            #net.load_state_dict(checkpoint["state_dict"])
            #net.update(force = True)
            #net.to(device) 
        print("sto allenando il modulo lrp: ",args.trainable_lrp)
        net.change_pars_lrp(tr = args.trainable_lrp)

        return net, baseline

    elif args.model == "latent":

        net = models[args.model](N = N, M = M, factorized_configuration = factorized_configuration, gaussian_configuration = gaussian_configuration, dim_adapter = args.dim_adapter)
        
        if args.pret_checkpoint is not None:
            state_dict = load_pretrained(torch.load(args.pret_checkpoint, map_location=device)['state_dict'])
            #print("faccio il check dei cumulative weights: ",net.gaussian_conditional.sos.cum_w)
            #print("prima di fare l'update abbiamo che: ",net.h_a[0].weight[0])
            del state_dict["gaussian_conditional._offset"] 
            del state_dict["gaussian_conditional._quantized_cdf"] 
            del state_dict["gaussian_conditional._cdf_length"] 
            del state_dict["gaussian_conditional.scale_table"] 
            del state_dict["entropy_bottleneck._offset"]
            del state_dict["entropy_bottleneck._quantized_cdf"]
            del state_dict["entropy_bottleneck._cdf_length"]
            net.load_state_dict(state_dict)
        net.to(device)       
        net.update()
        #print("***************************** CONTROLLO INOLTRE I CUMULATIVE WEIGHTS  ", net.gaussian_conditional.sos.cum_w)   
        baseline = False   
        return net, baseline



    else:
        net = models[args.model](N = N )
        baseline = True
        net = net.to(device)
        return net, baseline



def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",     
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):     # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )



