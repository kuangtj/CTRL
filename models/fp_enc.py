import torch
import torch.nn as nn
from models.mlp import MLP

class fp_encoder(nn.Module):
    def __init__(self, num_feature, hidden_dim = 256, compressed_dim=16):
        super().__init__()
        
        self.encoder = MLP(num_feature, [4 * hidden_dim], compressed_dim * hidden_dim)
        self.norm = nn.LayerNorm(compressed_dim * hidden_dim)
        self.compressed_dim = compressed_dim

    def forward(self, x):

        B = x.size(0)
        feature = self.norm(self.encoder(x.squeeze(dim=-1).float()))
        return feature.view(B, self.compressed_dim, -1)
