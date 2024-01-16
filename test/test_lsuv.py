import pytest
import torch
import torchvision as tv
from lsuv import lsuv_with_dataloader, lsuv_with_singlebatch


class TestLSUV:
    def test_vgg(self):
        device = torch.device('cpu')
        inp = torch.rand(8, 3, 128, 128)
        inp = inp / inp.std()
        model = tv.models.vgg11(pretrained=False)
        new_model = lsuv_with_singlebatch(model, inp, device=device)
    
    def test_vit(self):
        device = torch.device('cpu')
        inp = torch.rand(8, 3, 224, 224)
        inp = inp / inp.std()
        model = tv.models.vit_b_16(weights=None)
        new_model = lsuv_with_singlebatch(model, inp, device=device)

    def test_resnet(self):
        device = torch.device('cpu')
        inp = torch.rand(8, 3, 224, 224)
        inp = inp / inp.std()
        model = tv.models.resnet18(pretrained=False)
        new_model = lsuv_with_singlebatch(model, inp, device=device)
    
    def test_dataloader(self):
        device = torch.device('cpu')
        inp = torch.rand(8, 3, 224, 224)
        inp = inp / inp.std()
        model = tv.models.vgg11(weights=None)
        dataset = torch.utils.data.TensorDataset(inp)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        new_model = lsuv_with_dataloader(model, dataloader, device=device)
