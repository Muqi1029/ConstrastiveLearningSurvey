import torch
from torch import nn


class SimSiam(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # change the final fc module
        prev_dim = self.encoder.fc.weight.size(1) # 2048

        self.encoder.fc = nn.Sequential( # 3 layers MLP
            nn.Linear(prev_dim, prev_dim, bias=False), nn.BatchNorm1d(prev_dim), nn.ReLU(inplace=True), 
            nn.Linear(prev_dim, prev_dim, bias=False), nn.BatchNorm1d(prev_dim), nn.ReLU(inplace=True),
            self.encoder.fc, nn.BatchNorm1d(dim, affine=False)
        )
        self.encoder.fc[6].bias.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim), nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )
        
    
    def forward(self, x1, x2):
        """

        Args:
            x1 (batch_size, height, width): a view of a picture
            x2 (batch_size, height, width): another view of a picture
        """
        z1, z2 = self.encoder(x1), self.encoder(x2)

        p1, p2 = self.predictor(z2), self.predictor(z1)
        
        return p1, p2, z1.detach(), z2.detach() # stop grad
        

        

        

