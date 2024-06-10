import torch
from torch import Tensor, nn
from encoder import Encoder
from autoregressor import Autoregressor
from cpc.model.infoNCE import InfoNCE

class CPC(nn.Module):
    def __init__(self, 
                 args, 
                 strides, 
                 filter_sizes, 
                 padding, 
                 genc_input, 
                 genc_hidden,
                 gar_hidden):
        super().__init__()
        self.args = args
        self.encoder = Encoder(genc_input, genc_hidden, strides, filter_sizes, padding)

        self.autoregressor = Autoregressor(args, input_dim=genc_hidden, hidden_dim=gar_hidden)
        self.loss = InfoNCE(args) #TODO: INFO NCE
        
    
    def forward(self, x):
        z, c = self.get_latent_representations(x)
        loss, accuracy = self.loss.get(x, z, c)
        return loss, accuracy, z, c

    def get_latent_representations(self, x: Tensor):
        """Calculate latent representation of the input with encoder and autoregressor

        Args:
            x (Tensor): Shape: batch_size x input_dim x seq_len

        Returns:
            z - latent representation from the encoder (batch_size, seq_len, hidden_dim)
            c - latent representation of the autoregressor (batch_size, hidden_dim, seq_len)
        """
        z = self.encoder(x)
        z = z.permute(0, 2, 1)
        c = self.autoregressor(z)
        return z, c
        