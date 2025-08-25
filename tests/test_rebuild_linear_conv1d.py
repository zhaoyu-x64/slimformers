import torch
import torch.nn as nn
from transformers.modeling_utils import Conv1D

from slimformers import Pruner


def test_rebuild_linear_shapes_and_bias():
    lin = nn.Linear(7, 11, bias=True)
    pruner = Pruner(nn.Sequential())
    keep_out = torch.tensor([0, 2, 4, 6])
    keep_in = torch.tensor([1, 3, 5])
    new_lin = pruner._rebuild_linear(lin, keep_out=keep_out, keep_in=keep_in)

    assert new_lin.in_features == 3
    assert new_lin.out_features == 4
    assert new_lin.bias is not None

    assert torch.allclose(new_lin.weight, lin.weight.data[keep_out][:, keep_in])


def test_rebuild_conv1d_shapes():
    conv = Conv1D(nf=10, nx=6)
    pruner = Pruner(nn.Sequential())
    keep_out = torch.tensor([0, 3, 5])
    keep_in = torch.tensor([1, 4])
    new_conv = pruner._rebuild_conv1d(conv, keep_out=keep_out, keep_in=keep_in)

    assert new_conv.weight.shape == torch.Size([2, 3])
