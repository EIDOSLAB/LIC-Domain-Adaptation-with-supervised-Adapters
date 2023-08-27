import copy
import numpy as np 
from compress.entropy_models.weight_entropy_module import WeightEntropyModule
import re 
import torch 
import operator
#from compress.models.utils import encode_latent

class QuantizedModelWrapper:
    """Wrapper for Quantized Model"""

    def __init__(self, model,w_ent: WeightEntropyModule, regex: str) -> None:
        # regex: regex of keys to update
        self.model = copy.deepcopy(model)

        #self.model.modify_adapter(args,device)

        #for name,p in self.model.named_parameters():
          
            #p.requires_grad = False

        self.w_ent = w_ent
        self.regex = regex

        # self.params_init is defined here.
        self._register_params_init(model)

        # self.training is defined here.
        self.train()

    def train(self) -> None:
        self.training = True
        self.model.train()
        self.w_ent.train()

    def eval(self) -> None:
        self.training = False
        self.model.eval()
        self.w_ent.eval()

    def _register_params_init(self, model) -> None:
        params_dict = dict()
        for name, p in model.named_parameters():
            # encoder
            if name.startswith("g_a"):
                continue
            # hyper encoder
            if name.startswith("h_a"):
                continue

            params_dict[name] = p



        

        self.params_init: dict = copy.deepcopy(params_dict)


        for p in self.params_init.values():
            p.requires_grad = False

    

    def report_params(self):
        n_params_total: int = 0
        n_params_update: int = 0
        for key, p in self.params_init.items():
            n_param = np.prod(p.shape)
            n_params_total += n_param

            if re.match(self.regex, key) is not None:
                n_params_update += n_param



        print(f"#updating params/#total params: {n_params_update}/{n_params_total}")

    def __call__(self, x_pad: torch.Tensor, shape=None) -> dict:
        raise NotImplementedError("Please use models.helper instead.")

    def eval_enc(self):
        self.model.g_a.eval()
        self.model.h_a.eval()
        self.model.entropy_bottleneck.eval()
        self.model.gaussian_conditional.eval()

    
    def freeze_net(self):
        for name,p in self.model.named_parameters():
            p.requires_grad = False  


                 


    def train_adapter(self):
        self.model.pars_adapter(re_grad = True)

    def to(self, device):
        self.model.to(device)
        self.w_ent.to(device)
        for key in self.params_init.keys():
            self.params_init[key] = self.params_init[key].to(device)


    """
    @torch.no_grad()
    def compress(self, x_pad: torch.Tensor, y=None, z=None) -> dict:
        if y is not None and z is not None:
            compressed = encode_latent(self, y, z)
        else:
            compressed = self.model.compress(x_pad)
        compressed["weights"] = self.compress_weight()
        return compressed

    @torch.no_grad()
    def compress_weight(self) -> dict:
        weights = dict()
        for key, p_init in self.params_init.items():
            if re.match(self.regex, key) is not None:
                getter = operator.attrgetter(key)
                p_qua = getter(self.model)

                w_shape = p_init.reshape(1, 1, -1).shape
                diff = (p_qua - p_init).reshape(w_shape)
                weight = self.w_ent.compress(diff)
                weights[key] = weight
        return weights

    @torch.no_grad()
    def decompress(self, strings, shape, weights) -> dict:
        self.decompress_weight(weights)
        # out_dict has "x_hat" as a key.
        out_dict = self.model.decompress(strings, shape)
        return out_dict

    @torch.no_grad()
    def decompress_weight(self, weights: dict) -> None:
        for key, p_init in self.params_init.items():
            getter = operator.attrgetter(key)
            p_qua = getter(self.model)

            if key in weights.keys():
                weight = weights[key]
                diff = self.w_ent.decompress(weight, (p_init.numel(),))
                p_qua.copy_(p_init + diff.reshape(p_init.shape))

            else:
                p_qua.copy_(p_init)
    """
    def update_parameters(self, model) -> dict:
        """update model_qua parameters

        Args:
            model (CompressionModel): non-quantized model

        Returns:
            dict: m_likelihoods
        """
        # replace encoder params with the model one
        for p, p_qua in zip(model.parameters(), self.model.parameters()):
            p_qua.detach_()
            p_qua.copy_(p)

        # replace decoder params with the quantized one
        m_likelihoods = dict()
        for key, p_init in self.params_init.items():
            getter = operator.attrgetter(key)
            p = getter(model)
            p_qua = getter(self.model)

            # p_qua = p_init
            if re.match(self.regex, key) is None:
                p_qua.detach_()
                p_qua.copy_(p_init)
                m_likelihoods[key] = None

            # p_qua = p_init + diff_qua
            else:
                diff = p - p_init
                diff_qua, likelihood = self.w_ent(diff.reshape(1, 1, -1))
                p_new = p_init + diff_qua.reshape(diff.shape)
                p_qua.detach_()
                p_qua.copy_(p_new)
                m_likelihoods[key] = likelihood

        return m_likelihoods

    def update_ent(self, force: bool = False):
        self.model.update(force=force)
        self.w_ent.update(force=force)
        device = next(self.model.parameters()).device
        self.w_ent.to(device)