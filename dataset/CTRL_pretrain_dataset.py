import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset
from torch_scatter import scatter
from rdkit.Chem import BRICS
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, rdchem
from dataset.compound_constants import DAY_LIGHT_FG_SMARTS_LIST
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from dataset.get_pharmhgt_data import Mol2HeteroGraphData, Data2HeteroGraph
import concurrent.futures
import json

with open('desc_int_config.json', 'r', encoding='utf-8') as f:
    int_list = json.load(f)
        
def get_dataset(model_type=None):
    print("get_dataset model_type: ", model_type)
    train_dataset_list, valid_dataset_list = [], []
    from pathlib import Path
    print("loading data path......")
    
    pt_dir = Path("./data_store/pretrain_data_pt")
    all_data_item_path = sorted(["./data_store/pretrain_data_pt/" + p.name for p in pt_dir.glob("*.pt")])
    
    print("shuffle data path list......")
    import random

    random.shuffle(all_data_item_path)
    data_len = len(all_data_item_path)

    train_dataset_list = all_data_item_path[:int(0.95 * data_len)]
    valid_dataset_list = all_data_item_path[int(0.95 * data_len):]
    print("train_dataset_list: ", len(train_dataset_list))
    print("valid_dataset_list: ", len(valid_dataset_list))
        
    return train_dataset_list, valid_dataset_list


def compute_pairwise_metrics(pos, max_pairs=40000):
    N = pos.shape[0]
    if N < 2:
        return torch.zeros(0, 1), torch.zeros(0, 2, dtype=torch.long)

    idx_i, idx_j = torch.triu_indices(N, N, offset=1)

    dist_mat = torch.cdist(pos, pos, p=2)  # [N, N]
    dists = dist_mat[idx_i, idx_j]  # [M]
    
    if len(dists) > max_pairs:
        indices = torch.randperm(len(dists))[:max_pairs]
        dists = dists[indices]
        idx_i = idx_i[indices]
        idx_j = idx_j[indices]
    
    idx_pairs = torch.stack([idx_i, idx_j], dim=1)
    
    return dists.unsqueeze(-1), idx_pairs

def apply_valid_edge_mask_to_dgl(data_2D):
    eids_dict = {}

    for etype in data_2D.canonical_etypes:
        src_t, _, dst_t = etype
        src, dst = data_2D.edges(etype=etype)

        ns = data_2D.num_nodes(src_t)
        nd = data_2D.num_nodes(dst_t)

        mask = (src >= 0) & (dst >= 0) & (src < ns) & (dst < nd)
        eids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        eids_dict[etype] = eids

    data_2D = dgl.edge_subgraph(data_2D, eids_dict, preserve_nodes=True)
    return data_2D

    

from pathlib import Path   
class MoleculeDataset(Dataset):

    def __init__(self, dataset_list, model_type):
        super(Dataset, self).__init__()
        self.data_items = dataset_list
        self.model_type = model_type
        with open('desc_int_config.json', 'r', encoding='utf-8') as f:
            self.int_list = json.load(f)

    @staticmethod
    def get_item_dp(data_item, model_type):
        edge_attr = torch.cat([data_item['edge_attr'][:-len(data_item['x'])] + 1, data_item['edge_attr'][-len(data_item['x']):]], dim=0)

        data_2D = Data2HeteroGraph(data_item["fraginfo"])

        data_3D1 = Data(x=data_item['x'], pos=data_item['pos'], edge_index=data_item['edge_index'], edge_attr=edge_attr, bond_lengths=data_item['bond_lengths'])
        
        data_3D2 = Data(x=edge_attr, edge_index=data_item['super_edges'], edge_attr=data_item['bond_angles'])

        return data_2D, data_3D1, data_3D2, data_item
        
    def __getitem__(self, index):
        data_item = torch.load(self.data_items[index])
        p = Path(self.data_items[index])
        
        if (data_item['x'][:, 0].max() > 119 or data_item['x'][:, 0].min() < 0) or \
           (data_item['x'][:, 1].max() > 3 or data_item['x'][:, 1].min() < 0) or \
           (data_item['x'][:, 2].max() > 2 or data_item['x'][:, 2].min() < 0) or \
           (data_item['x'][:, 3].max() > 7 or data_item['x'][:, 3].min() < 0) or \
           (data_item['x'][:, 4].max() > 11 or data_item['x'][:, 4].min() < 0) or \
           (data_item['x'][:, 5].max() > 7 or data_item['x'][:, 5].min() < 0) or \
           (data_item['x'][:, 6].max() > 11 or data_item['x'][:, 6].min() < 0) or \
           (data_item['x'][:, 7].max() > 2 or data_item['x'][:, 7].min() < 0) or \
           (data_item['x'][:, 8].max() > 5 or data_item['x'][:, 8].min() < 0) or \
           (data_item['edge_attr'][:, 0].max() > 5 or data_item['edge_attr'][:, 0].min() < 0) or \
           (data_item['edge_attr'][:, 1].max() > 4 or data_item['edge_attr'][:, 1].min() < 0) or \
           (data_item['edge_attr'][:, 2].max() > 3 or data_item['edge_attr'][:, 2].min() < 0) or \
           (data_item['edge_attr'][:, 3].max() > 3 or data_item['edge_attr'][:, 3].min() < 0) or \
           (data_item['edge_attr'][:, 4].max() > 5 or data_item['edge_attr'][:, 4].min() < 0) or \
           data_item['edge_index'].max() + 1 != len(data_item['x']):
            next_index = (index + random.randrange(len(self))) % len(self)
            print("error file: ", self.data_items[index])
            return self.__getitem__(next_index)
        
        result = self.get_item_dp(data_item, self.model_type)
        data_2D, data_3D1, data_3D2, data_item_tmp = result 

        # --- sanitize edge_index for data_3D1 ---
        x = data_item['x']
        num_nodes = x.size(0)

        edge_index = data_item['edge_index']
        mask = torch.ones([data_item['edge_index'].shape[1]], dtype=bool)
        edge_index = edge_index[:, mask]
        data_item['edge_index'] = edge_index
        
        # src_mask
        max_nodes = 200
        D2, D3 = max_nodes, max_nodes
        Ddesc = 258
    
        L = D2 + D3 + Ddesc  # 总序列长度 658
        src_mask = torch.ones(L, L, dtype=torch.bool)
    
        adj2d = torch.zeros(D2, D2, dtype=torch.bool)
        src, dst = data_item_tmp['edge_index'][:, -len(data_item_tmp['pos'])]
        adj2d[src, dst] = True
        adj2d[dst, src] = True
        adj2d.fill_diagonal_(True)
        src_mask[:D2, :D2] = ~adj2d
    
        padded_pos = data_item_tmp['pos']
        dist = torch.cdist(padded_pos, padded_pos)
        adj3d = dist < 5.0
        adj3d.fill_diagonal_(True)
        src_mask[D2:D2 + padded_pos.size(0), D2:D2 + padded_pos.size(0)] = ~adj3d
        src_mask[D2:D2 + D3, D2:D2 + D3].fill_diagonal_(False)

        num_node = len(data_item_tmp['x'])
        src_mask[:num_node, D2+D3:] = False
        src_mask[D2:D2+num_node, D2+D3:] = False
        src_mask[D2+D3:, :num_node] = False
        src_mask[D2+D3:, D2:D2+num_node] = False

        selfloop = torch.ones(num_node, num_node, dtype=torch.bool)
        selfloop.fill_diagonal_(False)

        src_mask[:num_node, D2:D2+num_node] = selfloop
        src_mask[D2:D2+num_node, :num_node] = selfloop

        src_mask[D2+D3:, D2+D3:] = False
        
        # dist, idx_pairs
        dists, idx_pairs = compute_pairwise_metrics(data_item_tmp['pos'])
        
        # edge_attr_true, edge_index_true
        edge_attr_true = data_item_tmp['edge_attr'][0:-len(data_item_tmp['x']):2]
        edge_index_true = data_item_tmp['edge_index'][:, 0:-len(data_item_tmp['x']):2].transpose(1, 0)
        
        return data_2D, data_3D1, data_3D2, data_item_tmp['morgan_data'].unsqueeze(dim=-1), data_item_tmp['maccs_data'].unsqueeze(dim=-1), data_item_tmp['daylight_data'].unsqueeze(dim=-1), data_item_tmp['descriptor_int'].unsqueeze(dim=-1), data_item_tmp['descriptor_float'].unsqueeze(dim=-1), src_mask, data_item_tmp['mask_3d'], dists, idx_pairs, edge_attr_true, edge_index_true, data_item_tmp['smiles']

    def __len__(self):
        return len(self.data_items)


import dgl
from torch_geometric.data import Batch
class GeoPredCollateFn(object):
    """tbd"""
    def __init__(self, model_type, pretrain_tasks=None):
        self.pretrain_tasks = pretrain_tasks
        self.model_type = model_type
    
    def __call__(self, batch_data_list):
        """tbd"""
        data_2D_list = []
        data_3D1_list = []
        data_3D2_list = []
        morgan_list = []
        maccs_list = []
        daylight_list = []
        descriptor_int_list = []
        descriptor_float_list = []
        src_mask_list = []
        smiles_list = []
        mask_3d_list = []
        node_count = 0
        idx_pairs_list = []
        dist_list = []
        edge_attr_list = []
        edge_index_list = []

        for data_item in batch_data_list:

            data_2D_list.append(data_item[0])
            data_3D1_list.append(data_item[1])
            data_3D2_list.append(data_item[2])
            morgan_list.append(data_item[3].unsqueeze(0))
            maccs_list.append(data_item[4].unsqueeze(0))
            daylight_list.append(data_item[5].unsqueeze(0))
            descriptor_int_list.append(data_item[6].unsqueeze(0))
            descriptor_float_list.append(data_item[7].unsqueeze(0))
            src_mask_list.append(data_item[8].unsqueeze(0))
            mask_3d_list.append(data_item[9].unsqueeze(0))
            dist, idx_pairs, edge_attr, edge_index = data_item[10], data_item[11], data_item[12], data_item[13]
            idx_pairs += node_count
            edge_index += node_count
            idx_pairs_list.append(idx_pairs)
            dist_list.append(dist)
            edge_attr_list.append(edge_attr)
            edge_index_list.append(edge_index)
            node_count += len(data_item[1].x)
            smiles_list.append(data_item[14])

            
        data_2D_list = dgl.batch(data_2D_list)
        data_3D1_list = Batch.from_data_list(data_3D1_list)
        data_3D2_list = Batch.from_data_list(data_3D2_list)
        morgan_list = torch.cat(morgan_list, dim=0)
        maccs_list = torch.cat(maccs_list, dim=0)
        daylight_list = torch.cat(daylight_list, dim=0)
        descriptor_int_list = torch.cat(descriptor_int_list, dim=0)
        descriptor_float_list = torch.cat(descriptor_float_list, dim=0)
        src_mask_list = torch.cat(src_mask_list, dim=0)
        mask_3d_list = torch.cat(mask_3d_list, dim=0)
        idx_pairs_list = torch.cat(idx_pairs_list, dim=0)
        dist_list = torch.cat(dist_list, dim=0)
        edge_attr_list = torch.cat(edge_attr_list, dim=0)
        edge_index_list = torch.cat(edge_index_list, dim=0)
        
        return data_2D_list, data_3D1_list, data_3D2_list, morgan_list, maccs_list, daylight_list, descriptor_int_list, descriptor_float_list, src_mask_list, mask_3d_list, dist_list, idx_pairs_list, edge_attr_list, edge_index_list, smiles_list


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy('file_system')
            
class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers):
        super(object, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        print("MolTestDatasetWrapper init done")

    def get_data_loaders(self, model_type):
        train_dataset_list, valid_dataset_list = get_dataset(model_type)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset_list, valid_dataset_list, model_type)
        return train_loader, valid_loader
        
    def get_train_validation_data_loaders(self, train_dataset_list, valid_dataset_list, model_type):
        train_dataset, valid_dataset = MoleculeDataset(train_dataset_list, model_type), MoleculeDataset(valid_dataset_list, model_type)
        del train_dataset_list, valid_dataset_list
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=1, drop_last=True, shuffle=False, collate_fn=GeoPredCollateFn(model_type), pin_memory=True, timeout=0, worker_init_fn=set_worker_sharing_strategy)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=1, drop_last=True, shuffle=False, collate_fn=GeoPredCollateFn(model_type), pin_memory=True, timeout=0, worker_init_fn=set_worker_sharing_strategy)
        del train_dataset, valid_dataset
        return train_loader, valid_loader


        