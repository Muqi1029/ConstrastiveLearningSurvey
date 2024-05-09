from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, strides, filter_sizes, padding):
        super().__init__()

        assert (
            len(strides) == len(filter_sizes) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.net = nn.Sequential()
        for i, (s, f, p) in enumerate(zip(strides, filter_sizes, padding)):
            block = nn.Sequential(
                nn.Conv1d(in_channels=input_dim,
                          out_channels=hidden_dim, 
                          kernel_size=f,
                          stride=s,
                          padding=p),
                nn.ReLU()
            )
            self.net.add_module(f"layer-{i}", block)
            input_dim = hidden_dim
    
    def forward(self, x):
        return self.net(x)