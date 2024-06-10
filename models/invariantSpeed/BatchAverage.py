import torch
from torch import nn
from torch import Tensor


class BatchCriterion(nn.Module):
    def __init__(self, negM: float, T: float, batch_size: int):
        """

        Args:
            negM (float): the weight probability of negative samples
            T (float): the temperature of probability, which is used to centeralize the distribution 
            batch_size (int):
        """
        super().__init__()
        
        self.negM = negM 
        self.T = T
        self.batch_size = batch_size
        self.diag_mat = 1 - torch.eye(2 * batch_size).cuda()
    
    def forward(self, x: Tensor, targets) -> Tensor:
        """

        Args:
            x (Tensor): shape(2 * batch_size, feature_dim)
            targets (_type_): _description_

        Returns:
            Tensor: Binary Loss
        """
        pos = torch.exp(torch.sum(x[:self.batch_size] * x[self.batch_size:], dim=1) / self.T)
        
        # compute all prob except diagonal data
        all_prob = torch.exp((x @ x.T) / self.T) * self.diag_mat

        if self.negM == 1:
            all_div = torch.sum(all_div, dim=1)
        else:
            # remove pos for neg
            all_div = (torch.sum(all_div, dim=1) - pos) * self.negM + pos
        
        lnPmt = pos / all_div

        Pon_div = all_div.unsqueeze(dim=1)
        # Pon_div = torch.repeat_interleave(all_div, repeats=self.batch_size, dim=1)

        lnPon = all_prob / Pon_div
        
        # subtract the positive probability
        lnPon = torch.sum(torch.log(1 - lnPon), dim=1) - torch.log(1 - lnPmt)
        
        lnPmt = torch.log(lnPmt)
        
        lnPmtsum = torch.sum(lnPmt)
        lnPonsum = torch.sum(lnPon) * self.negM

        loss = - (lnPmtsum + lnPonsum) / self.batch_size
        return loss
