import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fp_enc import fp_encoder
from models.mlp import MLP
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter
from models.descriptor import DescriptorEmbeddingNet
from models.geognn import GeoGNNModel
import math
import dgl
import json

import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations).to(zis.device)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1).to(zis.device)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(zis.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


from models.pharmhgt import PharmHGT_AtomFuse
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 904):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, L, D = x.shape
        return x + self.pe[:L, :].unsqueeze(0)
        
        
class Mol_Encoder_norm(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
        device=None
    ):
        super().__init__()
        print("Mol_Encoder_norm")
        print("hidden_dim: ", hidden_dim)
        print("num_transformer_layers: ", num_transformer_layers)
        print("num_heads: ", num_heads)
        print("dropout: ", dropout)
        print("device: ", device)
        self.hidden_dim = hidden_dim

        # ——— 各模态编码器 ——— #
        self.encoder_frag  = PharmHGT_AtomFuse()
        self.encoder_3d    = GeoGNNModel(embed_dim=hidden_dim, layer_num=8, device=device)

        self.encoder_desc = DescriptorEmbeddingNet(int_emb_dim=hidden_dim)
        self.encoder_morgan = fp_encoder(num_feature=200, hidden_dim=hidden_dim)
        self.encoder_maccs = fp_encoder(num_feature=167, hidden_dim=hidden_dim)
        self.encoder_daylight = fp_encoder(num_feature=127, hidden_dim=hidden_dim)

        self.pe_2d = PositionalEncoding(hidden_dim, max_len=658)

        # ——— Transformer Encoder ——— #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*num_heads,
            dropout=0.2,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        self.pool = global_mean_pool
        self.norm = nn.LayerNorm(hidden_dim)
        self.lin_2d = nn.Linear(hidden_dim, hidden_dim)
        self.lin_3d = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, tasks=None):
        mask_2d = mask_3d
        batch_size = int(len(morgan_fp))

        frag_node, _ = self.encoder_frag(data_2D)
        feature_2d_tmp  = self.lin_2d(frag_node)
        mask_2d = mask_2d.reshape(int(batch_size), 200)
        feature_2d = torch.zeros(batch_size, 200, feature_2d_tmp.size(1), device=morgan_fp.device)
        feature_2d[~mask_3d] = feature_2d_tmp
        
        feature_3d_tmp  = self.lin_3d(self.encoder_3d(data_3D1, data_3D2)[0])
        mask_3d = mask_3d.reshape(int(batch_size), 200)
        feature_3d = torch.zeros(batch_size, 200, feature_3d_tmp.size(1), device=morgan_fp.device)
        feature_3d[~mask_3d] = feature_3d_tmp
        
        feature_desc_float, feature_desc_int = self.encoder_desc(descriptor_float, descriptor_int)
        feature_morgan  = self.encoder_morgan(morgan_fp)
        feature_maccs  = self.encoder_maccs(maccs_fp)
        feature_daylight  = self.encoder_daylight(daylight_fp)

        x = self.pe_2d(torch.cat([feature_2d, feature_3d, feature_desc_int, feature_desc_float, feature_morgan, feature_maccs, feature_daylight], dim=1))  
        src_mask = src_mask[:, :658, :658].unsqueeze(1).expand(-1, 8, -1, -1).reshape(-1, 658, 658)
        out = self.transformer(x, mask=src_mask)

        return out


class pretrain_model_norm(nn.Module):
    def __init__(self, hidden_dim: int = 256, tf_layer_num = 2, num_heads = 8, encoder=None, dropout=0.5, batch_size=192):
        super().__init__()
        self.enc = encoder
        print("pretrain_model_norm")
        print("hidden_dim: ", hidden_dim)
        print("tf_layer_num: ", tf_layer_num)
        print("num_heads: ", num_heads)
        print("dropout: ", dropout)

        self.nt_xent_criterion = NTXentLoss(batch_size, temperature=0.1, use_cosine_similarity=True)
        
        self.norm_1_2d = nn.LayerNorm(hidden_dim)
        self.norm_1_3d = nn.LayerNorm(hidden_dim)
        self.norm_1_morgan = nn.LayerNorm(hidden_dim)
        self.norm_1_macss = nn.LayerNorm(hidden_dim)
        self.norm_1_daylight = nn.LayerNorm(hidden_dim)
        self.norm_1_desc_int = nn.LayerNorm(hidden_dim)
        self.norm_1_desc_float = nn.LayerNorm(hidden_dim)
        
        self.pool = global_mean_pool
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 2d
        self.an = MLP(hidden_dim, [hidden_dim], 120 + 2)
        self.chi = MLP(hidden_dim, [hidden_dim], 4 + 2)
        self.aro = MLP(hidden_dim, [hidden_dim], 3 + 2)
        self.hyb = MLP(hidden_dim, [hidden_dim], 8 + 2)
        self.deg = MLP(hidden_dim, [hidden_dim], 8 + 2)
        self.hs = MLP(hidden_dim, [hidden_dim], 8 + 2)
        self.cha = MLP(hidden_dim, [hidden_dim], 10 + 2)
        self.iir = MLP(hidden_dim, [hidden_dim], 3 + 2)
        self.ele = MLP(hidden_dim, [hidden_dim], 5 + 2)
        self.bt = MLP(2 * hidden_dim, [hidden_dim], 6 + 2)
        self.bd = MLP(2 * hidden_dim, [hidden_dim], 5 + 2)
        self.bir = MLP(2 * hidden_dim, [hidden_dim], 4 + 2)
        self.bc = MLP(2 * hidden_dim, [hidden_dim], 4 + 2)
        self.bs = MLP(2 * hidden_dim, [hidden_dim], 6 + 2)
        self.loss_2d = nn.CrossEntropyLoss()
        
        # 3d
        self.bl = MLP(2 * hidden_dim, [hidden_dim], 1)
        self.loss_3d = nn.SmoothL1Loss()

        # fp
        self.fp_morgan = MLP(16 * hidden_dim, [2 * hidden_dim], 200)
        self.fp_maccs = MLP(16 * hidden_dim, [2 * hidden_dim], 167)
        self.fp_daylight = MLP(16 * hidden_dim, [2 * hidden_dim], 127)
        self.fp_loss = nn.SmoothL1Loss()

        with open('desc_int_config.json', 'r', encoding='utf-8') as f:
            self.int_list = json.load(f)
        with open('desc_float_config.json', 'r', encoding='utf-8') as f:
            self.float_list = json.load(f)
            
        self.desc_float_config = torch.load("./desc_float_config.pt")
            
        self.desc_int_layers = nn.ModuleList()
        for (v_max, v_min) in self.int_list:
            num_categories = v_max + 5
#            emb = nn.Embedding(num_categories, int_emb_dim)
            emb = MLP(hidden_dim, [int(hidden_dim/4)], num_categories)
#            print("num_categories: ", num_categories)
            # 为了直接用原始值作索引，我们后面会做： idx = val - v_min
            self.desc_int_layers.append(emb)

        self.desc_float_layers = nn.ModuleList()
        for (v_max, v_min) in self.float_list:
            emb = MLP(hidden_dim, [int(hidden_dim/4)], 1)
            # 为了直接用原始值作索引，我们后面会做： idx = val - v_min
            self.desc_float_layers.append(emb)
            
        self.desc_int_loss = nn.CrossEntropyLoss()
#        self.desc_int_loss = nn.MSELoss()
        self.desc_float_loss = nn.SmoothL1Loss()
        
        self.norm_2_2d = nn.LayerNorm(hidden_dim)
        self.norm_2_3d = nn.LayerNorm(hidden_dim)
        self.norm_2_morgan = nn.LayerNorm(hidden_dim)
        self.norm_2_macss = nn.LayerNorm(hidden_dim)
        self.norm_2_daylight = nn.LayerNorm(hidden_dim)
        self.norm_2_desc = nn.LayerNorm(hidden_dim)

    def mean_alignment_loss(self, reps_list):
        """
        最小化不同表征之间的平均距离
        """
        num_reps = len(reps_list)
        total_loss = 0
        count = 0
    
        for i in range(num_reps):
            for j in range(i + 1, num_reps):
                loss = self.nt_xent_criterion(reps_list[i], reps_list[j])
                total_loss += loss
                count += 1
    
        return total_loss / count if count > 0 else 0

    def forward(self, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, dist_list, idx_list, edge_attr, edge_index, step_n):
        out = self.enc(data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d)
        
        decoded = out

        rep_2d = self.norm_1_2d(decoded[:, :200][~mask_3d])
        rep_3d = self.norm_1_3d(decoded[:, 200:400][~mask_3d])
        rep_desc_int = self.norm_1_desc_int(decoded[:, 400:504])
        rep_desc_float = self.norm_1_desc_float(decoded[:, 504:610])
        rep_fp_morgan = self.norm_1_morgan(decoded[:, 610:626])
        rep_fp_maccs = self.norm_1_macss(decoded[:, 626:642])
        rep_fp_daylight = self.norm_1_daylight(decoded[:, 642:658])

        loss_dict = {}
        
        loss_dict['desc_int'] = 0
        for idx in range(descriptor_int.size(-1)):
            rep_desc_int_idx = rep_desc_int[:, idx]
            loss_dict['desc_int'] += self.desc_int_loss(self.desc_int_layers[idx](rep_desc_int_idx), descriptor_int[:, idx].squeeze(dim=1).long())
        loss_dict['desc_int'] = loss_dict['desc_int']/52

        loss_dict['desc_float'] = 0
        for idx in range(descriptor_float.size(-1)):
            rep_desc_float_idx = rep_desc_float[:, idx]
            loss_dict['desc_float'] += self.desc_float_loss(self.desc_float_layers[idx](rep_desc_float_idx), 5 * descriptor_float[:, idx]/(self.desc_float_config[idx][1] - self.desc_float_config[idx][0]))
        loss_dict['desc_float'] = loss_dict['desc_float']/53
        
        rec_fp_morgan = self.fp_morgan(rep_fp_morgan.reshape(len(rep_desc_int), -1))
        loss_dict['fp_morgan'] = self.fp_loss(rec_fp_morgan, morgan_fp.squeeze(dim=2).float())
        rec_fp_maccs = self.fp_maccs(rep_fp_maccs.reshape(len(rep_desc_int), -1))
        loss_dict['fp_maccs'] = self.fp_loss(rec_fp_maccs, maccs_fp.squeeze(dim=2).float())
        rec_fp_daylight = self.fp_daylight(rep_fp_daylight.reshape(len(rep_desc_int), -1))
        loss_dict['fp_daylight'] = self.fp_loss(rec_fp_daylight, daylight_fp.squeeze(dim=2).float())
        
        rec_an = self.an(rep_2d)
        rec_an_3d = self.an(rep_3d)
        loss_dict['an'] = self.loss_2d(rec_an, data_3D1.x[:, 0]) + self.loss_2d(rec_an_3d, data_3D1.x[:, 0])
        rec_chi = self.chi(rep_2d)
        rec_chi_3d = self.chi(rep_3d)
        loss_dict['chi'] = self.loss_2d(rec_chi, data_3D1.x[:, 1]) + self.loss_2d(rec_chi_3d, data_3D1.x[:, 1])
        rec_aro = self.aro(rep_2d)
        rec_aro_3d = self.aro(rep_3d)
        loss_dict['aro'] = self.loss_2d(rec_aro, data_3D1.x[:, 2]) + self.loss_2d(rec_aro_3d, data_3D1.x[:, 2])
        rec_hyb = self.hyb(rep_2d)
        rec_hyb_3d = self.hyb(rep_3d)
        loss_dict['hyb'] = self.loss_2d(rec_hyb, data_3D1.x[:, 3]) + self.loss_2d(rec_hyb_3d, data_3D1.x[:, 3])
        rec_deg = self.deg(rep_2d)
        rec_deg_3d = self.deg(rep_3d)
        loss_dict['deg'] = self.loss_2d(rec_deg, data_3D1.x[:, 4]) + self.loss_2d(rec_deg_3d, data_3D1.x[:, 4])
        rec_hs = self.hs(rep_2d)
        rec_hs_3d = self.hs(rep_3d)
        loss_dict['hs'] = self.loss_2d(rec_hs, data_3D1.x[:, 5]) + self.loss_2d(rec_hs_3d, data_3D1.x[:, 5])
        rec_cha = self.cha(rep_2d)
        rec_cha_3d = self.cha(rep_3d)
        loss_dict['cha'] = self.loss_2d(rec_cha, data_3D1.x[:, 6]) + self.loss_2d(rec_cha_3d, data_3D1.x[:, 6])
        rec_iir = self.iir(rep_2d)
        rec_iir_3d = self.iir(rep_3d)
        loss_dict['iir'] = self.loss_2d(rec_iir, data_3D1.x[:, 7]) + self.loss_2d(rec_iir_3d, data_3D1.x[:, 7])
        rec_ele = self.ele(rep_2d)
        rec_ele_3d = self.ele(rep_3d)
        loss_dict['ele'] = self.loss_2d(rec_ele, data_3D1.x[:, 8]) + self.loss_2d(rec_ele_3d, data_3D1.x[:, 8])
        
        b2d_feats = torch.cat([rep_2d[edge_index[:, 0]], rep_2d[edge_index[:, 1]]], dim=-1)
        b3d_feats = torch.cat([rep_2d[edge_index[:, 0]], rep_2d[edge_index[:, 1]]], dim=-1)
        b_feats = b2d_feats + b3d_feats
        
        rec_bt = self.bt(b_feats)
        loss_dict['bt'] = self.loss_2d(rec_bt, edge_attr[:, 0])

        rec_bd = self.bd(b_feats)
        loss_dict['bd'] = self.loss_2d(rec_bd, edge_attr[:, 1])

        rec_bir = self.bir(b_feats)
        loss_dict['bir'] = self.loss_2d(rec_bir, edge_attr[:, 2])

        rec_bc = self.bc(b_feats)
        loss_dict['bc'] = self.loss_2d(rec_bc, edge_attr[:, 3])

        rec_bs = self.bs(b_feats)
        loss_dict['bs'] = self.loss_2d(rec_bs, edge_attr[:, 4])

        rep_g = rep_3d + rep_2d
        
        bl_feats = torch.cat([rep_g[idx_list[:, 0]], rep_g[idx_list[:, 1]]], dim=-1)
        rec_bl = self.bl(bl_feats)
        loss_dict['bl'] = self.loss_3d(rec_bl, torch.clamp(dist_list, 0, 50))

        rep_2dg_mean = self.norm_2_2d(self.pool(rep_2d, data_3D1.batch))
        rep_3dg_mean = self.norm_2_3d(self.pool(rep_3d, data_3D1.batch))
        rep_desc_mean = self.norm_2_desc((rep_desc_int.mean(dim=1) + rep_desc_float.mean(dim=1))/2)
        rep_fp_morgan_mean = self.norm_2_morgan(rep_fp_morgan.mean(dim=1))
        rep_fp_maccs_mean = self.norm_2_macss(rep_fp_maccs.mean(dim=1))
        rep_fp_daylight_mean = self.norm_2_daylight(rep_fp_daylight.mean(dim=1))

        loss_dict['cll'] = self.mean_alignment_loss([rep_2dg_mean, rep_3dg_mean, rep_desc_mean, rep_fp_morgan_mean, rep_fp_maccs_mean, rep_fp_daylight_mean])

        total_loss = 0
        for name, raw_loss in loss_dict.items():
            total_loss = total_loss + raw_loss

        return total_loss, loss_dict