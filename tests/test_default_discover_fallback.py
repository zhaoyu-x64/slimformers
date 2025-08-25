import torch.nn as nn

from slimformers.discovery import discover_ffns_model_agnostic


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, x):
        return self.block(x)


def test_model_agnostic_discovers_pair():
    m = TinyMLP()
    blocks = discover_ffns_model_agnostic(m, min_hidden_dim=8)
    assert any(b["type"] == "ffn" for b in blocks)
    ffns = [b for b in blocks if b["type"] == "ffn"]
    assert ffns[0]["fc"] is not None and ffns[0]["proj"] is not None
