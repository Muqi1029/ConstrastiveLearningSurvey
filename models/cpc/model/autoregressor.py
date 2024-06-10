import torch
from torch import nn
from torch import Tensor


class Autoregressor(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.autoregressor = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): (batch_size, seq_len, input_dim)

        Returns:
            out: Tensor(batch_size, seq_len, hidden_dim)
        """
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        self.autoregressor.flatten_parameters()
        out, _ = self.autoregressor(x, h0)
        return out
