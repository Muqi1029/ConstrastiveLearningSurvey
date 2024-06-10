import pytest
from moco.builder import Moco
from torchvision import models
import torch


class TestMoco:
    @pytest.fixture(autouse=True)
    def pre_test(self):
        self.encoder_base = models.__dict__["resnet50"]
        self.batch_size = 32
        self.dim = 30
        self.K = 3200
    
    
    def test_model(self):
        moco = Moco(self.encoder_base, dim=self.dim, K=self.K)
        moco.eval()
        query = torch.randn(self.batch_size, 3, 224, 224)
        key = torch.randn(self.batch_size, 3, 224, 224)
        logits, labels = moco(query, key)
        assert logits.size() == (self.batch_size, self.K + 1) and labels.size() == (self.batch_size, )


    def test_vgg(self):
        pass

