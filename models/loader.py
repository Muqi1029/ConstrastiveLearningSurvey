from models.simCLR.model import ResNetSimCLR
from torch import nn


def load_model(model_name: str, **kwargs):
    if model_name == 'simclr':
        return ResNetSimCLR(base_model=kwargs['base_model'], out_dim=kwargs['out_dim'])
    else:
        raise NotImplementedError("model is not supported")
