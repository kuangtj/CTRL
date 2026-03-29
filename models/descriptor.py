import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.mlp import MLP
import json
class DescriptorEmbeddingNet(nn.Module):
    def __init__(self,
                 int_emb_dim: int = 256):
        super().__init__()
        with open('desc_float_config.json', 'r', encoding='utf-8') as f:
            float_list = json.load(f)
        with open('desc_int_config.json', 'r', encoding='utf-8') as f:
            int_list = json.load(f)
        centers = torch.tensor([c for c, g in float_list], dtype=torch.float32)
        gammas  = torch.tensor([g for c, g in float_list], dtype=torch.float32)
        self.register_buffer('centers', centers.view(1, -1, 20))
        self.log_gammas = nn.Parameter(torch.log(gammas).view(1, -1, 1))

        self.float_emb_layers = nn.ModuleList()
        for c, g in float_list:
            emb = nn.Linear(len(c), int_emb_dim)
            self.float_emb_layers.append(emb)

        # --- Embedding layers for each int descriptor ---
        self.int_emb_layers = nn.ModuleList()
        self.int_lin_emb_layers = nn.ModuleList()
        for (v_max, v_min) in int_list:
            num_categories = v_max + 5
            emb = nn.Embedding(num_categories, int_emb_dim)
            self.int_emb_layers.append(emb)
        self.int_mins = torch.tensor([v_min for v_min, v_max in int_list],
                                     dtype=torch.long)

    def forward(self,
                desc_float: torch.Tensor,
                desc_int:   torch.Tensor):

        B = desc_float.size(0)

        # --- RBF ---
        # Broadcast: (B,106,1) - (1,106,1) -> (B,106,1)
        diff = desc_float - self.centers
        # exp(-gamma * diff^2)
        rbf_feats = torch.exp(- torch.exp(self.log_gammas) * diff * diff)
        float_list = []
        for i, float_emb_layer in enumerate(self.float_emb_layers):
            float_list.append(float_emb_layer(rbf_feats[:, i]))
        float_embeds = torch.stack(float_list, dim=1)

        x_int = desc_int.squeeze(-1).long()  # [B,104]
        emb_list = []
        for i, emb_layer in enumerate(self.int_emb_layers):
            emb_list.append(emb_layer(x_int[:, i]))  # [B, int_emb_dim]
        # stack -> [B,104,int_emb_dim]
        int_embeds = torch.stack(emb_list, dim=1)

        return float_embeds, int_embeds

        