import torch
from torch import nn


class InfoNCE(nn.Module):
    def __init__(self, args, gar_hidden: int, genc_hidden: int):
        super().__init__()
        self.args = args
        
        self.gar_hidden = gar_hidden
        self.genc_hidden = genc_hidden
        self.negative_samples = self.args.negative_samples

        # predict | prediction_step | timesteps into the future
        self.predictor = nn.Linear(gar_hidden, genc_hidden * self.args.prediction_step, bias=False)
        
        self.loss = nn.LogSoftmax(dim=1)
    
    def get(self, x, z, c):
        full_z = z

        Wc = self.predictor(c)
        return self.infoNCE_loss(Wc, z, full_z)
    
    def infoNCE_loss(self, Wc, z, full_z):
        full_z = z
        return self.info
       
