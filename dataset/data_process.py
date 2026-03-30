import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader
from rdkit.Chem import BRICS
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, rdchem
from compound_constants import DAY_LIGHT_FG_SMARTS_LIST
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import yaml
from get_pharmhgt_data import Mol2HeteroGraphData, Data2HeteroGraph

def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
FORMAL_CHARGE_LIST = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
DEGREE_LIST = list(range(0, 11))
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]
HYBRIDIZATION_LIST = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.UNSPECIFIED
]

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        rdkit.Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
    'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values)
}

import matplotlib.pyplot as plt
from collections import Counter
from rdkit.Chem import Descriptors

def split_descriptor(descriptor):
    if True:
        bool_list = torch.tensor([False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])

        return torch.tensor(descriptor, dtype=torch.float)[bool_list], torch.tensor(descriptor, dtype=torch.float)[~bool_list]

def calculate_descriptors(mol):
    if True:
        """Calculate molecular descriptors."""
        data_tmp_list = []
        for desc_name, func in Descriptors.descList:
            try:
                value = func(mol)
            except Exception as e:
                value = None
                print(f"smiles: {smiles}, {desc_name} calculation error: {e}")
            data_tmp_list.append(value)
        return data_tmp_list
        
def get_maccs_fingerprint(mol):
    if True:
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]
        
day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]
def get_daylight_functional_group_counts(mol, fg_mo_list):
    if True:
        fg_counts = []
        for fg_mol in fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

from rdkit.Chem import AllChem, SDWriter
import torch
from rdkit import Chem
def pad_node_features(x, max_node_num=200):
    if True:
        """Pad node features to fixed size and create mask"""
        num_nodes = x.size(0)
    
        mask = torch.ones(max_node_num, dtype=torch.bool)
        mask[:num_nodes] = 0
    
        return mask

def get_bond_lengths(edges, atom_poses):
    if True:
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return torch.from_numpy(bond_lengths).float()

def get_superedge_angles(edges, atom_poses, dir_type='HT'):
    if True:
        """get superedge angles"""
        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        if E < 2:
            se = torch.zeros((0, 2), dtype=torch.long)
            ba = torch.zeros((0,), dtype=torch.float32)
            return se, ba
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0,], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return torch.from_numpy(super_edges).long(), torch.from_numpy(bond_angles).float()

def get_morgan_count_bitvector(mol, radius=2, nbits=200):
    fp = AllChem.GetHashedMorganFingerprint(
        mol,
        radius=radius,
        nBits=nbits
    )
    arr = [0] * nbits
    for bit_id, count in fp.GetNonzeroElements().items():
        arr[bit_id] = count
    return torch.tensor(arr)

from rdkit import Chem
from rdkit.Chem import AllChem
import torch

def save_mol_to_sdf(mol, best_cid, filename="best_conformation.sdf"):
    writer = Chem.SDWriter(filename)
    writer.write(mol, confId=best_cid)
    writer.close()
    
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

def get_molecule_positions(mol, sdf_file=None):
    
    mol_h = Chem.AddHs(mol) 
    
    cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=20, randomSeed=22, numThreads=0, useRandomCoords=True)
    
    best_cid = -1
    
    if not cids:
        mol_h.RemoveAllConformers() 
        AllChem.Compute2DCoords(mol_h)
        best_cid = 0 
        flag = False
        
    else:
        opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, maxIters=2000)
        
        energies = []
        valid_cids = []
        for i, (not_converged, energy) in enumerate(opt_results):
            energies.append(energy)
            valid_cids.append(cids[i])
        
        if not energies:
            mol_h.RemoveAllConformers()
            AllChem.Compute2DCoords(mol_h)
            best_cid = 0
            flag = False
            
        else:
            sortidx = torch.argsort(torch.tensor(energies))
            best_list_idx = int(sortidx[0])
            best_cid = valid_cids[best_list_idx]
            flag = True

    mol_no_h = Chem.RemoveHs(mol_h)

    final_pos = mol_no_h.GetConformer(best_cid).GetPositions()
    pos = torch.tensor(final_pos, dtype=torch.float)

    if flag and sdf_file is not None:
        save_mol_to_sdf(mol_no_h, best_cid, filename=sdf_file)
        
    return pos, flag


def smiles_preporcess(smiles_tmp, sdf_file=None, pretrain=False):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tmp))
    data = {}
    data['smiles'] = smiles
    mol = Chem.MolFromSmiles(smiles)
    data['morgan_data'] = torch.tensor(get_morgan_count_bitvector(mol), dtype=torch.float)
    data['maccs_data'] = torch.tensor(get_maccs_fingerprint(mol), dtype=torch.float)
    data['daylight_data'] = torch.tensor(get_daylight_functional_group_counts(mol, day_light_fg_mo_list), dtype=torch.float)
    descriptor = calculate_descriptors(mol)
    data['descriptor_int'], data['descriptor_float'] = split_descriptor(descriptor)


    # graph
    atomic_num, chirality, is_aromatic, hybridization, degree, num_hs, formal_charge, is_in_ring, num_radical_electrons = [], [], [], [], [], [], [], [], []
    for atom in mol.GetAtoms():
        atomic_num.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        chirality.append(x_map['chirality'].index(atom.GetChiralTag()))
        is_aromatic.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        hybridization.append(x_map['hybridization'].index(atom.GetHybridization()))
        degree.append(x_map['degree'].index(atom.GetDegree()))
        num_hs.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        formal_charge.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        is_in_ring.append(x_map['is_in_ring'].index(atom.IsInRing()))
        num_radical_electrons.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))

    x1 = torch.tensor(atomic_num, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality, dtype=torch.long).view(-1,1)
    x3 = torch.tensor(is_aromatic, dtype=torch.long).view(-1,1)
    x4 = torch.tensor(hybridization, dtype=torch.long).view(-1,1)
    x5 = torch.tensor(degree, dtype=torch.long).view(-1,1)
    x6 = torch.tensor(num_hs, dtype=torch.long).view(-1,1)
    x7 = torch.tensor(formal_charge, dtype=torch.long).view(-1,1)
    x8 = torch.tensor(is_in_ring, dtype=torch.long).view(-1,1)
    x9 = torch.tensor(num_radical_electrons, dtype=torch.long).view(-1,1)

    x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9], dim=-1)
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir()),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
            e_map['bond_stereo'].index(bond.GetStereo())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir()),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
            e_map['bond_stereo'].index(bond.GetStereo())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    data['x'] = x
    data['edge_index'] = edge_index
    data['edge_attr'] = edge_attr

    data['mask_3d'] = pad_node_features(data['x'])


    # pos
    if sdf_file is None:
        pos, flag = get_molecule_positions(mol, sdf_file)
        data['pos'] = pos
        data['3d_flag'] = flag
    elif os.path.isfile(sdf_file):
        suppl = Chem.SDMolSupplier(sdf_file)
        mol_pos = suppl[0]
        if mol_pos is None:
            raise ValueError(f"Fail to read mol from {sdf_file}")
        conf = mol_pos.GetConformer()
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos_i = conf.GetAtomPosition(i)
            positions.append([pos_i.x, pos_i.y, pos_i.z])
        pos = torch.tensor(positions, dtype=torch.float)
        data['pos'] = pos
        data['3d_flag'] = True
    else:
        pos, flag = get_molecule_positions(mol, sdf_file)
        data['pos'] = pos
        data['3d_flag'] = flag

    # src_mask
    if not pretrain:
        max_nodes = 200
        D2, D3 = max_nodes, max_nodes
        Ddesc = 258
    
        L = D2 + D3 + Ddesc
        src_mask = torch.ones(L, L, dtype=torch.bool)
    
        adj2d = torch.zeros(D2, D2, dtype=torch.bool)
        src, dst = data['edge_index']
        adj2d[src, dst] = True
        adj2d[dst, src] = True
        adj2d.fill_diagonal_(True)
        src_mask[:D2, :D2] = ~adj2d
    
        pos = data['pos']  # [N_i, 3]
        padded_pos = torch.zeros(D3, 3, device=pos.device)
        padded_pos[:pos.size(0)] = pos
        dist = torch.cdist(padded_pos, padded_pos)
        adj3d = dist < 5.0
        adj3d.fill_diagonal_(True)
        src_mask[D2:D2+D3, D2:D2+D3] = ~adj3d

        num_node = len(data['x'])
        src_mask[:num_node, D2+D3:] = False
        src_mask[D2:D2+num_node, D2+D3:] = False
        src_mask[D2+D3:, :num_node] = False
        src_mask[D2+D3:, D2:D2+num_node] = False

        selfloop = torch.ones(num_node, num_node, dtype=torch.bool)
        selfloop.fill_diagonal_(False)

        src_mask[:num_node, D2:D2+num_node] = selfloop
        src_mask[D2:D2+num_node, :num_node] = selfloop

        src_mask[D2+D3:, D2+D3:] = False

        data['src_mask'] = src_mask

    # self_loop
    num_nodes = data['x'].shape[0]
    if data['edge_index'].numel() == 0:
        edge_attr_dim = 5
        loop_index = torch.zeros((2, num_nodes), 
                               dtype=torch.long,
                               device=data['edge_attr'].device)
        loop_attr = torch.zeros((num_nodes, edge_attr_dim), 
                               dtype=torch.long,
                               device=data['edge_attr'].device)
        data['edge_index'] = loop_index
        data['edge_attr'] = loop_attr
    else:
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=data['edge_index'].device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)  # [2, num_nodes]

        loop_attr  = torch.zeros((num_nodes, data['edge_attr'].size(1)), dtype=data['x'].dtype, device=data['x'].device)

        data['edge_index'] = torch.cat([data['edge_index'], loop_index], dim=1)
        data['edge_attr']  = torch.cat([data['edge_attr'],  loop_attr ], dim=0)

    # data3D2
    data['bond_lengths'] = get_bond_lengths(data['edge_index'].transpose(0, 1), data['pos']).unsqueeze(-1)
    super_edges, bond_angles = get_superedge_angles(data['edge_index'].transpose(0, 1), data['pos'])
    bond_angles = bond_angles.unsqueeze(-1)
    data['super_edges'] = super_edges.transpose(0, 1)
    data['bond_angles'] = bond_angles

    num_nodes = data['edge_attr'].shape[0]
    if data['super_edges'].numel() == 0:
        edge_attr_dim = 1
        loop_index = torch.zeros((2, num_nodes), 
                               dtype=torch.long,
                               device=data['bond_angles'].device)
        loop_attr = torch.zeros((num_nodes, edge_attr_dim), 
                               dtype=torch.long,
                               device=data['bond_angles'].device)
        data['super_edges'] = loop_index
        data['bond_angles'] = loop_attr
    else:
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=data['super_edges'].device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)  # [2, num_nodes]

        loop_attr  = torch.zeros((num_nodes, data['bond_angles'].size(1)), dtype=data['x'].dtype, device=data['x'].device)

        data['super_edges'] = torch.cat([data['super_edges'], loop_index], dim=1)
        data['bond_angles']  = torch.cat([data['bond_angles'],  loop_attr ], dim=0)

    data['descriptor_float'] = data['descriptor_float'].squeeze(dim=-1)
    data['descriptor_int'] = data['descriptor_int'].squeeze(dim=-1)
    data['morgan_data'] = data['morgan_data'].squeeze(dim=-1)
    data['maccs_data'] = data['maccs_data'].squeeze(dim=-1)
    data['daylight_data'] = data['daylight_data'].squeeze(dim=-1)

    data["fraginfo"] = Mol2HeteroGraphData(Chem.MolFromSmiles(data['smiles']))
    if not pretrain:
        data["fraggraph"] = Data2HeteroGraph(data["fraginfo"])

    return data

    
import os
import io
import json
import argparse
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import esm

import os
import torch
from typing import Optional, Dict, Any

import esm

class ESM2_3DEncoder:
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        
        load_fn = getattr(esm.pretrained, model_name)
        self.model, self.alphabet = load_fn()
        self.model = self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        
        try:
            self.repr_layer = int(model_name.split('_')[1][1:])
        except:
            self.repr_layer = self.model.num_layers

    @torch.no_grad()
    def encode(self, seq: str) -> torch.Tensor:
        data = [("protein", seq)]
        # batch_labels, batch_strs, batch_tokens
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)  
        
        results = self.model(
            batch_tokens,
            repr_layers=[self.repr_layer],
            return_contacts=False,
        )
        
        # [B, T, d]
        token_reprs = results["representations"][self.repr_layer]  
        tokens_len = batch_lens[0].item()
        
        per_residue = token_reprs[0]  
            
        return per_residue.cpu()


def get_esm2_3d_repr_from_pdb(
    prot_seq: str,
    encoder: Optional[ESM2_3DEncoder] = None,
    device: str = "cuda",
    esm2_model_name: str = "esm2_t33_650M_UR50D",
) -> torch.Tensor:
    if encoder is None:
        encoder = ESM2_3DEncoder(model_name=esm2_model_name, device=device)
    
    seq_emb = encoder.encode(prot_seq)
    return seq_emb


def get_pocket_feat_esm2_only(prot_seq: str, device: str = "cuda:0", encoder: Optional[ESM2_3DEncoder] = None) -> torch.Tensor:
    esm2_out = get_esm2_3d_repr_from_pdb(
        prot_seq,
        encoder=encoder,
        device=device,
    )
    esm2_out = esm2_out.unsqueeze(dim=0) 
    return esm2_out 


def prot_preporcess(prot_seq: str, device: str = "cuda:0", encoder: Optional[ESM2_3DEncoder] = None) -> torch.Tensor:
    esm2_out = get_pocket_feat_esm2_only(prot_seq, device=device, encoder=encoder)
    return esm2_out