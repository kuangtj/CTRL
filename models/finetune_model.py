import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fp_enc import fp_encoder
from models.mlp import MLP
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter
from models.descriptor import DescriptorEmbeddingNet
from models.geognn import GeoGNNModel
from models.pharmhgt import PharmHGT_AtomFuse
import math
import dgl


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 658):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
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
        num_transformer_layers: int = 6,
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

        self.desc_int_att = MLP(hidden_dim, [hidden_dim], 1, dropout_rate=0.2)
        self.desc_float_att = MLP(hidden_dim, [hidden_dim], 1, dropout_rate=0.2)
        self.morgan_att = MLP(hidden_dim, [hidden_dim], 1, dropout_rate=0.2)
        self.maccs_att = MLP(hidden_dim, [hidden_dim], 1, dropout_rate=0.2)
        self.daylight_att = MLP(hidden_dim, [hidden_dim], 1, dropout_rate=0.2)
        
        self.norm_1_2d = nn.LayerNorm(hidden_dim)
        self.norm_1_3d = nn.LayerNorm(hidden_dim)
        self.norm_1_morgan = nn.LayerNorm(hidden_dim)
        self.norm_1_macss = nn.LayerNorm(hidden_dim)
        self.norm_1_daylight = nn.LayerNorm(hidden_dim)
        self.norm_1_desc_int = nn.LayerNorm(hidden_dim)
        self.norm_1_desc_float = nn.LayerNorm(hidden_dim)

    def forward(self, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d):
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

        rep_2d = out[:, :200][~mask_2d]
        rep_3d = out[:, 200:400][~mask_3d]
        w_desc_int= torch.softmax(self.desc_int_att(out[:, 400:504]), dim=1)  # [B, L, 1]
        rep_desc_int = (out[:, 400:504] * w_desc_int).sum(dim=1)
        w_desc_float= torch.softmax(self.desc_float_att(out[:, 504:610]), dim=1)  # [B, L, 1]
        rep_desc_float = (out[:, 504:610] * w_desc_float).sum(dim=1)
        w_morgan= torch.softmax(self.morgan_att(out[:, 610:626]), dim=1)  # [B, L, 1]
        rep_fp_morgan = (out[:, 610:626] * w_morgan).sum(dim=1)
        w_maccs= torch.softmax(self.maccs_att(out[:, 626:642]), dim=1)  # [B, L, 1]
        rep_fp_maccs = (out[:, 626:642] * w_maccs).sum(dim=1)
        w_daylight= torch.softmax(self.daylight_att(out[:, 642:658]), dim=1)  # [B, L, 1]
        rep_fp_daylight = (out[:, 642:658] * w_daylight).sum(dim=1)

        rep_2dg = self.pool(rep_2d, data_3D1.batch)
        rep_3dg = scatter(rep_3d, data_3D1.batch, dim=0)

        return torch.cat([self.norm_1_2d(rep_2dg), self.norm_1_3d(rep_3dg), self.norm_1_desc_int(rep_desc_int), self.norm_1_desc_float(rep_desc_float), self.norm_1_morgan(rep_fp_morgan), self.norm_1_macss(rep_fp_maccs), self.norm_1_daylight(rep_fp_daylight)], dim=-1)


class finetune_model_norm_all_task_together(nn.Module):
    def __init__(self, hidden_dim: int = 256, tf_layer_num=2, num_heads=8,
                 tasks_info=None, encoder=None):
        super().__init__()
        print("finetune_model_norm_all_task_together")
        print("hidden_dim: ", hidden_dim)
        print("tf_layer_num: ", tf_layer_num)
        print("num_heads: ", num_heads)
        print("tasks_info: ", tasks_info)

        self.enc = encoder

        # ---------- Task Embedding ----------
        self.task_ids = list(tasks_info.keys())                
        self.task_id2idx = {tid: i for i, tid in enumerate(self.task_ids)}
        self.task_embed = nn.Embedding(len(self.task_ids), 7 * hidden_dim)

        # ---------- coop attention ----------
        self.sample_coop_attn = nn.MultiheadAttention(
            embed_dim=7 * hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.sample_coop_ln = nn.LayerNorm(7 * hidden_dim)
        self.coop_attn_norm1 = nn.LayerNorm(7 * hidden_dim)
        self.coop_attn_norm2 = nn.LayerNorm(7 * hidden_dim)

        self.pred_heads = nn.ModuleDict()
        self.pred_heads_norm = nn.ModuleDict()
        self.pred_heads_prot_norm = nn.ModuleDict()
        self.pred_heads_cellline_norm = nn.ModuleDict()
        self.pred_heads_weightinteg = nn.ModuleDict()
        self.pred_heads_ryp_norm_0 = nn.ModuleDict()
        self.pred_heads_ryp_norm_1 = nn.ModuleDict()
        self.pred_heads_ryp_norm_2 = nn.ModuleDict()
        self.pred_head_prot_feat_t = nn.ModuleDict()
        self.pred_head_prot = nn.ModuleDict()
        self.pred_head_default_c = nn.ParameterDict()
        self.pred_head_default_c_lin = nn.ModuleDict()
        self.pred_head_cellline_mlp_drp = nn.ModuleDict()
        self.pred_head_cellline_mlp_dsp = nn.ModuleDict()
        self.pred_heads_ryp_weightinteg_0 = nn.ModuleDict()
        self.pred_heads_ryp_weightinteg_1 = nn.ModuleDict()
        self.pred_heads_ryp_weightinteg_2 = nn.ModuleDict()
        self.pred_heads_weightinteg = nn.ModuleDict()

        self.pred_heads_task_emb = nn.ParameterDict()
        for task_id, sub_task in tasks_info.items():
            self.pred_heads_task_emb[task_id] = nn.Parameter(torch.zeros((7 * hidden_dim)))

        # pred head
        for task_id, sub_task in tasks_info.items():
            self.pred_heads_weightinteg[task_id] = MLP(hidden_dim, [hidden_dim, hidden_dim], 1)
            if task_id in ['bbbp', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp2c9_veith', 'cyp2c9_substrate_carbonmangels']:
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(7 * hidden_dim, [int(2.5 * hidden_dim)], len(sub_task)),
                    nn.Sigmoid()
                )
            if task_id in ['hiv', 'hia_hou', 'cyp3a4_substrate_carbonmangels', 'herg', 'ames', 'dili', 'cyp2d6_veith', 'cyp3a4_veith', 'cyp2d6_substrate_carbonmangels', 'clintox', 'sider']:
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(7 * hidden_dim, [int(2.5 * hidden_dim), hidden_dim], len(sub_task)),
                    nn.Sigmoid()
                )
            if task_id in ['esol', 'lipophilicity', 'lipophilicity_astrazeneca', 'ppbr_az', 'ld50_zhu', 'vdss_lombardo', 'half_life_obach']:
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(7 * hidden_dim, [int(2.5 * hidden_dim)], len(sub_task))
                )
            if task_id in ['caco2_wang', 'solubility_aqsoldb', 'clearance_hepatocyte_az', 'clearance_microsome_az']:
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(7 * hidden_dim, [int(2.5 * hidden_dim), hidden_dim], len(sub_task))
                )
            if task_id in ['davis', 'kiba', 'bindingdb_kd']:
                self.pred_head_prot_feat_t[task_id] = MLP(1280, [int(2.5 * hidden_dim)], hidden_dim)
                pred_head_encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.5,
                    batch_first=True
                )
                self.pred_head_prot[task_id] = nn.TransformerEncoder(pred_head_encoder_layer, num_layers=2)
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(8 * hidden_dim, [int(2.5 * hidden_dim), hidden_dim], 1),
                )
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads_prot_norm[task_id] = nn.LayerNorm(1 * hidden_dim)
            if task_id in ['buchwald-hartwig']:
                self.pred_head_default_c[task_id] = nn.Parameter(torch.zeros((7 * hidden_dim)))
                self.pred_head_default_c_lin[task_id] = nn.Linear(7 * hidden_dim, 7 * hidden_dim)
                self.pred_heads_ryp_norm_0[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads_ryp_norm_1[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads_ryp_norm_2[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads_ryp_weightinteg_0[task_id] = MLP(hidden_dim, [hidden_dim, hidden_dim], 1)
                self.pred_heads_ryp_weightinteg_1[task_id] = MLP(hidden_dim, [hidden_dim, hidden_dim], 1)
                self.pred_heads_ryp_weightinteg_2[task_id] = MLP(hidden_dim, [hidden_dim, hidden_dim], 1)
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(21 * hidden_dim, [int(2.5 * hidden_dim), hidden_dim], 1),
                    nn.Sigmoid()
                )
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
            if task_id in ['drugbank']:
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(14 * hidden_dim, [int(2.5 * hidden_dim), hidden_dim], 86),
                    nn.Sigmoid()
                )
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
            if task_id in ['gdsc1', 'gdsc2']:
                self.pred_head_cellline_mlp_drp[task_id] = MLP(17737, [4 * hidden_dim, 2 * hidden_dim], hidden_dim)
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(8 * hidden_dim, [int(2.5 * hidden_dim), hidden_dim], 1)
                )
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads_cellline_norm[task_id] = nn.LayerNorm(1 * hidden_dim)
            if task_id in ['oncopolypharmacology']:
                self.pred_head_cellline_mlp_dsp[task_id] = MLP(8785, [4 * hidden_dim, 2 * hidden_dim], hidden_dim)
                self.pred_heads[task_id] = nn.Sequential(
                    MLP(15 * hidden_dim, [int(2.5 * hidden_dim), hidden_dim], 1)
                )
                self.pred_heads_norm[task_id] = nn.LayerNorm(7 * hidden_dim)
                self.pred_heads_cellline_norm[task_id] = nn.LayerNorm(1 * hidden_dim)
              
        self.attn_threshold = 0.0
        self.T = 2.0          
        self.alpha = 0.3      

    def _filter_valid_tasks(self, tasks):
        valid_task_list = []
        for t in tasks:
            if t in ['ames', 'bbb_martins', 'bioavailability_ma', 'cyp2c9_substrate_carbonmangels', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith', 'cyp3a4_substrate_carbonmangels', 'cyp3a4_veith', 'dili', 'herg', 'hia_hou', 'pgp_broccatelli', 'caco2_wang', 'clearance_hepatocyte_az', 'clearance_microsome_az', 'lipophilicity', 'half_life_obach', 'ld50_zhu', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb', 'vdss_lombardo', 'bbbp', 'clintox', 'sider', 'hiv', 'esol', 'lipophilicity', 'davis', 'kiba', 'bindingdb_kd', 'gdsc1', 'gdsc2']:
                valid_task_list.append(t)
            if t in ['drugbank', 'oncopolypharmacology']:
                valid_task_list.append(t)
                valid_task_list.append(t)
            if t in ['buchwald-hartwig']:
                valid_task_list.append(t)
                valid_task_list.append(t)
                valid_task_list.append(t)
        return valid_task_list

    def forward(self, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, epoch_counter):

        rep = self.enc(data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d)  # [B, 7*hidden_dim]

        valid_task_list = self._filter_valid_tasks(tasks)
        assert len(valid_task_list) == len(rep), f"rep rows {len(rep)} != valid_task_list {len(valid_task_list)}"

        N, D = rep.shape

        joint_rep = torch.empty_like(rep)  # [N, D]
        for i, t in enumerate(valid_task_list):
            joint_rep[i] = rep[i] + self.pred_heads_task_emb[t]

        if epoch_counter < 6:
            x = joint_rep.unsqueeze(0)  # [1, B, D]
            attn_out, attn_w = self.sample_coop_attn(x, x, x, need_weights=True, average_attn_weights=False)
            attn_out = self.coop_attn_norm1(attn_out)
            rep_coop = attn_out.squeeze(0)
            rep_final = joint_rep + 0.25 * rep_coop
        else:
            rep_final = joint_rep
            
        pre_label = []
        rep_final = rep_final.view(N, 7, -1)
        
        mol_idx, prot_idx, prot_start_idx, cellline_drp_idx, cellline_dsp_idx, ryp_idx = 0, 0, 0, 0, 0, 0
        for idx in range(len(tasks)):
            if tasks[idx] in ['ames', 'bbb_martins', 'bioavailability_ma', 'cyp2c9_substrate_carbonmangels', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith', 'cyp3a4_substrate_carbonmangels', 'cyp3a4_veith', 'dili', 'herg', 'hia_hou', 'pgp_broccatelli', 'caco2_wang', 'clearance_hepatocyte_az', 'clearance_microsome_az', 'lipophilicity', 'half_life_obach', 'ld50_zhu', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb', 'vdss_lombardo', 'bbbp', 'clintox', 'sider', 'hiv', 'esol', 'lipophilicity']:
                # mol
                rep_tmp_i = rep_final[mol_idx]
                logits_tmp = self.pred_heads_weightinteg[tasks[idx]](rep_tmp_i)
                weight_tmp = torch.softmax(logits_tmp / self.T, dim=0)
                rep_tmp = (rep_tmp_i * weight_tmp).reshape(-1) + rep_final[mol_idx].reshape(-1)
                # pre_label
                pre_label.append(self.pred_heads[tasks[idx]](rep_tmp))
                mol_idx += 1
            if tasks[idx] in ['davis', 'kiba', 'bindingdb_kd']:
                # mol
                rep_tmp_i = rep_final[mol_idx]
                logits_tmp = self.pred_heads_weightinteg[tasks[idx]](rep_tmp_i)
                weight_tmp = torch.softmax(logits_tmp / self.T, dim=0)
                rep_tmp = (rep_tmp_i * weight_tmp).reshape(-1) + rep_final[mol_idx].reshape(-1)
                # prot
                prot_rep_i_tmp = prot_rep[:, prot_start_idx:prot_start_idx + prot_num[prot_idx]]
                prot_rep_i = self.pred_head_prot_feat_t[tasks[idx]](prot_rep_i_tmp)
                prot_rep_i_1 = self.pred_head_prot[tasks[idx]](prot_rep_i).mean(dim=1)
                # pre_label
                pre_label.append(self.pred_heads[tasks[idx]](torch.cat([rep_tmp, self.pred_heads_prot_norm[tasks[idx]](prot_rep_i_1.squeeze(0))], dim=-1)))
                prot_start_idx = prot_start_idx + prot_num[prot_idx]
                mol_idx += 1
                prot_idx += 1
            if tasks[idx] in ['drugbank']:
                # mol_0
                rep_tmp_i_0 = rep_final[mol_idx]
                logits_tmp_0 = self.pred_heads_weightinteg[tasks[idx]](rep_tmp_i_0)
                weight_tmp_0 = torch.softmax(logits_tmp_0 / self.T, dim=0)
                rep_tmp_0 = (rep_tmp_i_0 * weight_tmp_0).reshape(-1) + rep_final[mol_idx].reshape(-1)
                # mol_1
                rep_tmp_i_1 = rep_final[mol_idx + 1]
                logits_tmp_1 = self.pred_heads_weightinteg[tasks[idx]](rep_tmp_i_1)
                weight_tmp_1 = torch.softmax(logits_tmp_1 / self.T, dim=0)
                rep_tmp_1 = (rep_tmp_i_1 * weight_tmp_1).reshape(-1) + rep_final[mol_idx + 1].reshape(-1)
                # pre_label
                pre_label.append(self.pred_heads[tasks[idx]](torch.cat([rep_tmp_0, rep_tmp_1], dim=-1)))
                mol_idx += 2
            if tasks[idx] in ['buchwald-hartwig']:  
                # mol_0
                rep_tmp_i_0 = rep_final[mol_idx]
                logits_tmp_0 = self.pred_heads_ryp_weightinteg_0[tasks[idx]](rep_tmp_i_0)
                weight_tmp_0 = torch.softmax(logits_tmp_0 / self.T, dim=0)
                rep_tmp_0 = (rep_tmp_i_0 * weight_tmp_0).reshape(-1) + rep_final[mol_idx].reshape(-1)
                # mol_1
                rep_tmp_i_1 = rep_final[mol_idx + 1]
                logits_tmp_1 = self.pred_heads_ryp_weightinteg_1[tasks[idx]](rep_tmp_i_1)
                weight_tmp_1 = torch.softmax(logits_tmp_1 / self.T, dim=0)
                rep_tmp_1 = (rep_tmp_i_1 * weight_tmp_1).reshape(-1) + rep_final[mol_idx + 1].reshape(-1)
                if is_catalyst[ryp_idx] == True:
                    rep_tmp_i_2 = rep_final[mol_idx + 2]
                    logits_tmp_2 = self.pred_heads_ryp_weightinteg_2[tasks[idx]](rep_tmp_i_2)
                    weight_tmp_2 = torch.softmax(logits_tmp_2 / self.T, dim=0)
                    rep_tmp_2 = (rep_tmp_i_2 * weight_tmp_2).reshape(-1) + rep_final[mol_idx + 2].reshape(-1)
                else:
                    catalyst_rep_i_tmp = self.pred_head_default_c[tasks[idx]]
                    rep_tmp_2 = self.pred_head_default_c_lin[tasks[idx]](catalyst_rep_i_tmp)
                pre_label.append(self.pred_heads[tasks[idx]](torch.cat([rep_tmp_0, rep_tmp_1, rep_tmp_2], dim=-1)))
                mol_idx += 3
                ryp_idx += 1
            if tasks[idx] in ['gdsc1', 'gdsc2']:
                # mol_0
                rep_tmp_i = rep_final[mol_idx]
                logits_tmp = self.pred_heads_weightinteg[tasks[idx]](rep_tmp_i)
                weight_tmp = torch.softmax(logits_tmp / self.T, dim=0)
                rep_tmp = (rep_tmp_i * weight_tmp).reshape(-1) + rep_final[mol_idx].reshape(-1)
                # cellline
                rep_cellline_drp_i = self.pred_head_cellline_mlp_drp[tasks[idx]](cellline_drp_rep[cellline_drp_idx])
                # pre_label
                pre_label.append(self.pred_heads[tasks[idx]](torch.cat([rep_tmp, self.pred_heads_cellline_norm[tasks[idx]](rep_cellline_drp_i)], dim=-1)))
                cellline_drp_idx += 1
                mol_idx += 1
            if tasks[idx] in ['oncopolypharmacology']:
                # mol_0
                rep_tmp_i_0 = rep_final[mol_idx]
                logits_tmp_0 = self.pred_heads_weightinteg[tasks[idx]](rep_tmp_i_0)
                weight_tmp_0 = torch.softmax(logits_tmp_0 / self.T, dim=0)
                rep_tmp_0 = (rep_tmp_i_0 * weight_tmp_0).reshape(-1) + rep_final[mol_idx].reshape(-1)
                # mol_1
                rep_tmp_i_1 = rep_final[mol_idx + 1]
                logits_tmp_1 = self.pred_heads_weightinteg[tasks[idx]](rep_tmp_i_1)
                weight_tmp_1 = torch.softmax(logits_tmp_1 / self.T, dim=0)
                rep_tmp_1 = (rep_tmp_i_1 * weight_tmp_1).reshape(-1) + rep_final[mol_idx + 1].reshape(-1)
                # cellline
                rep_cellline_dsp_i = self.pred_head_cellline_mlp_dsp[tasks[idx]](cellline_dsp_rep[cellline_dsp_idx])
                # pre_label
                pre_label.append(self.pred_heads[tasks[idx]](torch.cat([rep_tmp_0, rep_tmp_1, self.pred_heads_cellline_norm[tasks[idx]](rep_cellline_dsp_i)], dim=-1)))
                cellline_dsp_idx += 1
                mol_idx += 2
        
        assert mol_idx == len(rep)
        if len(prot_num) != 0:
            assert prot_idx == len(prot_num)
            assert prot_start_idx == prot_rep.shape[1]
            assert prot_start_idx == sum(prot_num)
        if len(is_catalyst) != 0:
            assert ryp_idx == len(is_catalyst)
        if len(cellline_drp_rep) != 0:
            assert cellline_drp_idx == len(cellline_drp_rep)
        if len(cellline_dsp_rep) != 0:
            assert cellline_dsp_idx == len(cellline_dsp_rep)

        return pre_label





