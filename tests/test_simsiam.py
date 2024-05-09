from torchvision import models
from simSiam.builder import SimSiam
import pytest


class TestModel:
    @pytest.fixture(autouse=True)
    def pre_test(self):
        dim = 128
        pred_dim = 1024
        base_encoder = models.__dict__['resnet50']
        self.model = SimSiam(base_encoder, dim, pred_dim)
        print(self.model)
 
 
    def test_model(self):
        pass
        
    def test_res(self):
        pass
