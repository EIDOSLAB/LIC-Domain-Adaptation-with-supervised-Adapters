
import torch
import torch.nn as nn
from compress.zoo import models
from compressai.ans import BufferedRansEncoder, RansDecoder
from compress.ops import ste_round
import copy 



def rename_key_for_adapter(key, stringa, nuova_stringa):
    if key.startswith(stringa):
        key = nuova_stringa  + key[6:]
    return key








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


def get_model_for_evaluation(args,model_path,device):
    if args.model == "base":

        print("attn block base method siamo in baseline")

        checkpoint = torch.load(model_path, map_location=device)#["state_dict"]


        if "entropy_bottleneck._quantized_cdf" in list(checkpoint.keys()):
            del checkpoint["entropy_bottleneck._offset"]
            del checkpoint["entropy_bottleneck._quantized_cdf"]
            del checkpoint["entropy_bottleneck._cdf_length"]
        if "gaussian_conditional._quantized_cdf" in list(checkpoint.keys()):
            del checkpoint["gaussian_conditional._offset"]
            del checkpoint["gaussian_conditional._quantized_cdf"]
            del checkpoint["gaussian_conditional._cdf_length"]
            del checkpoint["gaussian_conditional.scale_table"]
            
            

                
        print("INIZIO STATE DICT")
        net = from_state_dict(models[args.model], checkpoint)

        net.update()
        net.to(device) 


        return net
    elif args.model == "latent":
        checkpoint = load_pretrained(torch.load(model_path, map_location=device))
        state_dict = checkpoint["state_dict"]
        checkpoint["N"] = 192
        checkpoint["M"] = 320
        net = models[args.model](N = checkpoint["N"],
                                M = checkpoint["M"],
                              #  factorized_configuration = checkpoint["factorized_configuration"], 
                                gaussian_configuration = checkpoint["gaussian_configuration"]) #, dim_adapter = args.dim_adapter)

        if "gaussian_conditional._quantized_cdf" in list(state_dict.keys()):
            del state_dict["gaussian_conditional._offset"] 
            del state_dict["gaussian_conditional._quantized_cdf"] 
            del state_dict["gaussian_conditional._cdf_length"] 
            del state_dict["gaussian_conditional.scale_table"] 
        if "entropy_bottleneck._quantized_cdf" in list(state_dict.keys()):
            del state_dict["entropy_bottleneck._offset"]
            del state_dict["entropy_bottleneck._quantized_cdf"]
            del state_dict["entropy_bottleneck._cdf_length"]
            
        net.load_state_dict(state_dict)
        net.to(device)       
        net.update()
        #print("***************************** CONTROLLO INOLTRE I CUMULATIVE WEIGHTS  ", net.gaussian_conditional.sos.cum_w)   
    
        return net




def get_model(args,device, N = 192, M = 320 ) -> nn.Module:

    if args.model == "base":
        print("attn block base method siamo in baseline")
        if args.pret_checkpoint is not None: 

            print("entroa qua per la baseline!!!!")
            #net.update(force = True)
            checkpoint = torch.load(args.pret_checkpoint, map_location=device)#["state_dict"]

            """
            if "entropy_bottleneck._quantized_cdf" in list(checkpoint.keys()):
                del checkpoint["entropy_bottleneck._offset"]
                del checkpoint["entropy_bottleneck._quantized_cdf"]
                del checkpoint["entropy_bottleneck._cdf_length"]
            if "gaussian_conditional._quantized_cdf" in list(checkpoint.keys()):
                del checkpoint["gaussian_conditional._offset"]
                del checkpoint["gaussian_conditional._quantized_cdf"]
                del checkpoint["gaussian_conditional._cdf_length"]
                del checkpoint["gaussian_conditional.scale_table"]
            """
            

                
            print("INIZIO STATE DICT")
            net = from_state_dict(models[args.model], checkpoint)

            net.update()
            net.to(device)
        else: 
            net = models[args.model](N = N ,M = M)



        return net

    elif args.model in ("latent","rate"):

        net = models[args.model](N = N, M = M) #, dim_adapter = args.dim_adapter)
        
        if args.pret_checkpoint is not None:

            #state_dict = load_pretrained(torch.load(args.pret_checkpoint, map_location=device)['state_dict'])
            #print("-------> ",state_dict.keys())
            state_dict = load_pretrained(torch.load(args.pret_checkpoint, map_location=device))
            #print("faccio il check dei cumulative weights: ",net.gaussian_conditional.sos.cum_w)#jj
            #print("prima di fare l'update abbiamo che: ",net.h_a[0].weight[0])
            if "gaussian_conditional._quantized_cdf" in list(state_dict.keys()):
                del state_dict["gaussian_conditional._offset"] 
                del state_dict["gaussian_conditional._quantized_cdf"] 
                del state_dict["gaussian_conditional._cdf_length"] 
                del state_dict["gaussian_conditional.scale_table"] 
            if "entropy_bottleneck._quantized_cdf" in list(state_dict.keys()):
                del state_dict["entropy_bottleneck._offset"]
                del state_dict["entropy_bottleneck._quantized_cdf"]
                del state_dict["entropy_bottleneck._cdf_length"]
            
        net.load_state_dict(state_dict)
        net.to(device)       
        net.update()
        #print("***************************** CONTROLLO INOLTRE I CUMULATIVE WEIGHTS  ", net.gaussian_conditional.sos.cum_w) # ffff  fff
  
        return net

    elif args.model in ("decoder"):


        checkpoint = torch.load(args.pret_checkpoint , map_location=device)#["state_dict"]

            
        print("INIZIO STATE DICT")
        modello_base = from_state_dict(models["base"], checkpoint)

        #print("questa è la dimensione iniziale: ",checkpoint["g_a.4.conv_b.0.attn.relative_position_bias_table"].shape)




        



        modello_base.update()
        modello_base.to(device) 

        net = models[args.model](N = modello_base.N,
                                M =modello_base.M,
                                dim_adapter_1 = args.dim_adapter_1,
                                dim_adapter_2 = args.dim_adapter_2,
                                stride_1 = args.stride_1,
                                stride_2 = args.stride_2,
                                bias = args.bias,
                                kernel_size_1 = args.kernel_size_1,
                                kernel_size_2 = args.kernel_size_2,
                                padding_1 = args.padding_1,
                                padding_2 = args.padding_2,
                                std = args.std,
                                mean = args.mean,
                                position = args.position,
                                type_adapter = args.type_adapter
                              ) 
        
        #print("questo è il nuovo modello: ",net.state_dict()["g_a.4.conv_b.0.attn.relative_position_bias_table"].shape)

        
        state_dict = modello_base.state_dict()
        state_dict = {rename_key_for_adapter(k, stringa = "g_s.8.", nuova_stringa = "g_s.9." ): v for k, v in state_dict.items()}
        state_dict = {rename_key_for_adapter(k, stringa = "g_s.7.",nuova_stringa = "g_s.8."): v for k, v in state_dict.items()}
        #state_dict = rename_key_for_adapter(state_dict)


        info = net.load_state_dict(state_dict, strict=False)
        net.to(device)

        """
        print("***************************************************************************************************")
        print("****************************** MODELLO NUOVO ********************************************")
        print("***************************************************************************************************")

        for k in list(net.state_dict().keys()):
            if "g_s" in k:
                print(k)
        print("******************************************************************************************************")
        print("******************************************************************************************************")
        print("******************************************************************************************************")
        """


        return net



    else:
        net = models[args.model](N = N )
        baseline = True
        net = net.to(device)
        return net, baseline


def introduce_teacher(args, device):
    teacher_net = models["base"](N = args.dims_n, M = args.dims_m)
    checkpoint = torch.load(args.pret_checkpoint_teacher, map_location=device)
    if "entropy_bottleneck._quantized_cdf" in list(checkpoint.keys()):
        del checkpoint["entropy_bottleneck._offset"]
        del checkpoint["entropy_bottleneck._quantized_cdf"]
        del checkpoint["entropy_bottleneck._cdf_length"]
    if "gaussian_conditional._quantized_cdf" in list(checkpoint.keys()):
        del checkpoint["gaussian_conditional._offset"]
        del checkpoint["gaussian_conditional._quantized_cdf"]
        del checkpoint["gaussian_conditional._cdf_length"]
        del checkpoint["gaussian_conditional.scale_table"]
                     
    teacher_net.load_state_dict(checkpoint)
    teacher_net.to(device) 
    teacher_net.update()
    return teacher_net
    


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



