import torch 
from torch import Tensor, nn


class Moco(nn.Module):
    def __init__(self, base_encoder, dim=30, K=4000, m=0.999, T=0.07, mlp=False):
        """Initialize Moco

        Args:
            base_encoder (_type_): _description_
            dim (int, optional): feature dimension. Defaults to 30.
            K (int, optional): queue size; number of negative keys. Defaults to 4000.
            m (float, optional): moco momentum. Defaults to 0.999.
            T (float, optional): temperature. Defaults to 0.07.
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    

    def forward(self, im_q, im_k):
        """

        Args:
            im_q (Tensor): batch_size x C
            im_k (_type_): batch_size x C
        """
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()

            # TODO: shuffle for making use of BN
    
            k = self.encoder_k(im_k) # N x dim
            k = nn.functional.normalize(k, dim=1)

            # TODO: unshuffle

        l_pos = torch.sum(q * k, dim=1, keepdim=True)
        l_neg = q @ self.queue

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long)

        self._dequeue_and_enqueue(k)
        return logits, labels

    @torch.no_grad() 
    def _dequeue_and_enqueue(self, keys: Tensor):
        """enqueue k into queue and dequeue the oldest keys

        Args:
            k (Tensor): N x dim
        """
        batch_size = keys.size(0)
        ptr = self.queue_ptr.item()
        
        assert self.K % batch_size == 0 
        self.queue[:, ptr: ptr + batch_size] = keys.T
        
        self.queue_ptr[0] = (ptr + batch_size) % self.K
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for q_param, k_param in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k_param = self.m * k_param.data + (1 - self.m) * q_param.data

