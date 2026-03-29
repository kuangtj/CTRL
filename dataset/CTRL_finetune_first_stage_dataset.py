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

def get_morgan_fingerprint_long(mol, radius=2):
    if True:
        """Get Morgan fingerprint."""
        nBits = 2048
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return torch.tensor([int(b) for b in mfp.ToBitString()])

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
import concurrent.futures
import json
from rdkit.Chem import Descriptors

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold



def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, item in enumerate(dataset):
        smiles = item['smiles']
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    train_dataset, valid_dataset, test_dataset = [], [], []
    for i in train_inds:
        train_dataset.append(dataset[i])
    for i in valid_inds:
        valid_dataset.append(dataset[i])
    for i in test_inds:
        test_dataset.append(dataset[i])
    return train_dataset, valid_dataset, test_dataset

def pad_node_features(x, max_node_num=200):
    if True:
        """Pad node features to fixed size and create mask"""
        num_nodes = x.size(0)
    
        mask = torch.ones(max_node_num, dtype=torch.bool)
        mask[:num_nodes] = 0
    
        return mask

def split_descriptor(descriptor):
    if True:
        bool_list = torch.tensor([False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
        return torch.tensor(descriptor, dtype=torch.float)[bool_list], torch.tensor(descriptor, dtype=torch.float)[~bool_list]
        
with open('desc_int_config.json', 'r', encoding='utf-8') as f:
    int_list = json.load(f)

with open('./data/DTI_data_from_TDC/dti_smiles_dict.json', 'r', encoding='utf-8') as f:
    dti_smiles_dict = json.load(f)

with open('./data/DTI_data_from_TDC/dti_prot_dict.json', 'r', encoding='utf-8') as f:
    dti_prot_dict = json.load(f)

from collections import defaultdict

def merge_drug_interactions(Drug_0_list, Drug_1_list, label_list, max_label, min_label):
   
    drug_pair_dict = defaultdict(set)
    
    for drug_0, drug_1, label in zip(Drug_0_list, Drug_1_list, label_list):
        drug_pair = tuple(sorted([drug_0, drug_1]))
        drug_pair_dict[drug_pair].add(label)

    merged_drug_pairs = []
    merged_labels = []
    
    for drug_pair, labels in drug_pair_dict.items():
        merged_drug_pairs.append(drug_pair)
        
        binary_vector = [0] * (max_label - min_label + 1)
        for label in labels:
            binary_vector[label - min_label] = 1
        
        merged_labels.append(binary_vector)

    return merged_drug_pairs, merged_labels
    
with open('./data/DSP_data_from_TDC/dsp_smiles_list.json', 'r', encoding='utf-8') as f:
    dsp_smiles_list = json.load(f)

with open('./data/DRP_data_from_TDC/drp_smiles_dict.json', 'r', encoding='utf-8') as f:
    drp_smiles_dict = json.load(f)

with open('./data/DDI_data_from_TDC/ddi_smiles_dict.json', 'r', encoding='utf-8') as f:
    ddi_smiles_dict = json.load(f)

with open('./data/RYP_data_from_TDC/ryp_smiles_dict.json', 'r', encoding='utf-8') as f:
    ryp_smiles_dict = json.load(f)

import copy
def deep_copy_tensor_dict(tensor_dict):
    return {key: value.clone() for key, value in tensor_dict.items()}
        
def get_dataset(model_type=None):
    print("get_dataset model_type: ", model_type)
    train_dataset_list, valid_dataset_list, test_dataset_list = [], [], []
    task_id_list = [
        "hia_hou",
        "pgp_broccatelli",
        "bioavailability_ma",
        "bbb_martins",
        "ames",
        "dili",
        "caco2_wang",
        "lipophilicity_astrazeneca",
        "solubility_aqsoldb",
        "ppbr_az",
        "ld50_zhu",
        "vdss_lombardo",
        "half_life_obach",
        "clearance_hepatocyte_az",
        "clearance_microsome_az",
        "bbbp",
        "clintox",
        "esol",
        "lipophilicity",
        "sider",
        "hiv",
        "bindingdb_kd",
        "davis",
        "kiba",
        "drugbank",
        "buchwald-hartwig",
        "gdsc1",
        "gdsc2",
        "oncopolypharmacology",
    ]
    from tdc.single_pred import ADME, Tox, Yields
    from tdc.multi_pred import DTI, DDI, DrugRes, DrugSyn
    for task_id in task_id_list:
        print("task_id: ", task_id)

        # ADMET TDC
        if task_id in ["hia_hou", "pgp_broccatelli", "bioavailability_ma", "bbb_martins", "vdss_lombardo", "cyp3a4_substrate_carbonmangels", "herg", "ames", "dili", "cyp2d6_veith", "cyp3a4_veith", "cyp2c9_veith", "cyp2d6_substrate_carbonmangels", "cyp2c9_substrate_carbonmangels", "caco2_wang", "solubility_aqsoldb", "lipophilicity_astrazeneca", "ppbr_az", "vdss_lombardo", "half_life_obach", "clearance_hepatocyte_az", "ld50_zhu", "clearance_microsome_az"]:
            train_data_items_pro = torch.load(f'./data_store/train_' + task_id + '.pt')
            valid_data_items_pro = torch.load(f'./data_store/valid_' + task_id + '.pt')
            test_data_items_pro = torch.load(f'./data_store/test_' + task_id + '.pt')

        # ADMET MoleculeNet
        if task_id in ["bbbp", "clintox", "esol", "lipophilicity", "sider", "hiv"]:
            train_data_items_pro = torch.load(f'./data_store/train_' + task_id + '.pt')
            valid_data_items_pro = torch.load(f'./data_store/valid_' + task_id + '.pt')
            test_data_items_pro = torch.load(f'./data_store/test_' + task_id + '.pt')

        # DTI TDC
        if task_id in ["bindingdb_kd", "davis", "kiba"]:
            
            frac = [0.9, 0.02, 0.08]
#            frac = [0.98, 0.01, 0.01]
            if task_id == "davis":
                data = DTI(name = 'DAVIS', path="./data/DTI_data_from_TDC")
                data.convert_to_log(form = 'binding')
                split = data.get_split(method = 'cold_split', column_name = 'Drug', frac = frac)
            if task_id == "kiba":
                data = DTI(name = 'KIBA', path="./data/DTI_data_from_TDC")
                split = data.get_split(method = 'cold_split', column_name = 'Drug', frac = frac)
            if task_id == "bindingdb_kd":
                data = DTI(name = 'BindingDB_Kd', path="./data/DTI_data_from_TDC")
                data.harmonize_affinities(mode = 'max_affinity')
                data.convert_to_log(form = 'binding')
                split = data.get_split(method = 'cold_split', column_name = 'Drug', frac = frac)

            train_smiles_list = split['train'].iloc[:, 1].tolist()
            valid_smiles_list = split['valid'].iloc[:, 1].tolist()
            test_smiles_list = split['test'].iloc[:, 1].tolist()

            train_molname_list = split['train'].iloc[:, 0].tolist()
            valid_molname_list = split['valid'].iloc[:, 0].tolist()
            test_molname_list = split['test'].iloc[:, 0].tolist()

            train_pn_list = split['train'].iloc[:, 2].tolist()
            valid_pn_list = split['valid'].iloc[:, 2].tolist()
            test_pn_list = split['test'].iloc[:, 2].tolist()

            train_ps_list = split['train'].iloc[:, 3].tolist()
            valid_ps_list = split['valid'].iloc[:, 3].tolist()
            test_ps_list = split['test'].iloc[:, 3].tolist()

            train_labels_list = split['train'].iloc[:, 4].tolist()
            valid_labels_list = split['valid'].iloc[:, 4].tolist()
            test_labels_list = split['test'].iloc[:, 4].tolist()
        
            train_data_items_list = []
            for idx in range(len(train_smiles_list)):
                if not os.path.exists(f"./data_store/DTI_mol_pt/mol_{str(train_molname_list[idx]) + str(dti_smiles_dict[train_smiles_list[idx]].replace('/', '_')[:245 - len(str(train_molname_list[idx]))])}.pt"):
                    print("error train data path: ", f"./data_store/DTI_mol_pt/mol_{str(train_molname_list[idx]) + str(dti_smiles_dict[train_smiles_list[idx]].replace('/', '_')[:245 - len(str(train_molname_list[idx]))])}.pt")
                    continue
                if not os.path.exists(f"./data_store/DTI_prot_pt/{dti_prot_dict[test_ps_list[idx]]}.pt"):
                    print("error train data path: ", f"./data_store/DTI_prot_pt/{dti_prot_dict[test_ps_list[idx]]}.pt")
                    continue
                item = {}
                item['protein_rep'] = f"./data_store/DTI_prot_pt/{dti_prot_dict[train_ps_list[idx]]}.pt"
                item['mol_pt'] = f"./DTI_data/DTI_mol_data_pt_file_frag_true/data_{str(train_molname_list[idx]) + str(dti_smiles_dict[train_smiles_list[idx]].replace('/', '_')[:245 - len(str(train_molname_list[idx]))])}.pt"
                item['label'] = train_labels_list[idx]
                item['task_id'] = task_id
                train_data_items_list.append(item)
            
            valid_data_items_list = []
            for idx in range(len(valid_smiles_list)):
                if not os.path.exists(f"./data_store/DTI_mol_pt/mol_{str(valid_molname_list[idx]) + str(dti_smiles_dict[valid_smiles_list[idx]].replace('/', '_')[:245 - len(str(valid_molname_list[idx]))])}.pt"):
                    print("error valid data path: ", f"./data_store/DTI_mol_pt/mol_{str(valid_molname_list[idx]) + str(dti_smiles_dict[valid_smiles_list[idx]].replace('/', '_')[:245 - len(str(valid_molname_list[idx]))])}.pt")
                    continue
                if not os.path.exists(f"./data_store/DTI_prot_pt/{dti_prot_dict[test_ps_list[idx]]}.pt"):
                    print("error valid data path: ", f"./data_store/DTI_prot_pt/{dti_prot_dict[test_ps_list[idx]]}.pt")
                    continue
                item = {}
                item['protein_rep'] = f"./data_store/DTI_prot_pt/{dti_prot_dict[valid_ps_list[idx]]}.pt"
                item['mol_pt'] = f"./data_store/DTI_mol_pt/mol_{str(valid_molname_list[idx]) + str(dti_smiles_dict[valid_smiles_list[idx]].replace('/', '_')[:245 - len(str(valid_molname_list[idx]))])}.pt"
                item['label'] = valid_labels_list[idx]
                item['task_id'] = task_id
                valid_data_items_list.append(item)
            
            test_data_items_list = []
            for idx in range(len(test_smiles_list)):
                if not os.path.exists(f"./data_store/DTI_mol_pt/mol_{str(test_molname_list[idx]) + str(dti_smiles_dict[test_smiles_list[idx]].replace('/', '_')[:245 - len(str(test_molname_list[idx]))])}.pt"):
                    print("error test data path: ", f"./data_store/DTI_mol_pt/mol_{str(test_molname_list[idx]) + str(dti_smiles_dict[test_smiles_list[idx]].replace('/', '_')[:245 - len(str(test_molname_list[idx]))])}.pt")
                    continue
                if not os.path.exists(f"./data_store/DTI_prot_pt/{dti_prot_dict[test_ps_list[idx]]}.pt"):
                    print("error test data path: ", f"./data_store/DTI_prot_pt/{dti_prot_dict[test_ps_list[idx]]}.pt")
                    continue
                item = {}
                item['protein_rep'] = f"./data_store/DTI_prot_pt/{dti_prot_dict[test_ps_list[idx]]}.pt"
                item['mol_pt'] = f"./data_store/DTI_mol_pt/mol_{str(test_molname_list[idx]) + str(dti_smiles_dict[test_smiles_list[idx]].replace('/', '_')[:245 - len(str(test_molname_list[idx]))])}.pt"
                item['label'] = test_labels_list[idx]
                item['task_id'] = task_id
                test_data_items_list.append(item)
                
            train_data_items_pro, valid_data_items_pro, test_data_items_pro = train_data_items_list, valid_data_items_list, test_data_items_list

        
        # DDI TDC
        if task_id in ["drugbank"]:
            if task_id == 'drugbank':
                max_label = 86
                min_label = 1
                DDI_name = 'DrugBank'
            from tdc.multi_pred import DDI
            data = DDI(name = DDI_name, path="./data/DDI_data_from_TDC")

            train_data_items_list = []
            valid_data_items_list = []
            test_data_items_list = []

            split = data.get_split()
            
            label_0_list = split['train'].iloc[:, 4].tolist()
            data0_0_list = split['train'].iloc[:, 3].tolist()
            data1_0_list = split['train'].iloc[:, 1].tolist()
            merged_drug_pairs_0, merged_labels_0 = merge_drug_interactions(data0_0_list, data1_0_list, label_0_list, max_label=max_label, min_label=min_label)
            
            label_1_list = split['valid'].iloc[:, 4].tolist()
            data0_1_list = split['valid'].iloc[:, 3].tolist()
            data1_1_list = split['valid'].iloc[:, 1].tolist()
            merged_drug_pairs_1, merged_labels_1 = merge_drug_interactions(data0_1_list, data1_1_list, label_1_list, max_label=max_label, min_label=min_label)
            
            label_2_list = split['test'].iloc[:, 4].tolist()
            data0_2_list = split['test'].iloc[:, 3].tolist()
            data1_2_list = split['test'].iloc[:, 1].tolist()
            merged_drug_pairs_2, merged_labels_2 = merge_drug_interactions(data0_2_list, data1_2_list, label_2_list, max_label=max_label, min_label=min_label)
        
            for data_idx in range(len(merged_labels_0)):
                item = {}
                item['task_id'] = task_id
                if not os.path.exists('./data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_0[data_idx][0]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_0[data_idx][0]].replace('/', '_')[:245] + '.pt')
                    continue
                if not os.path.exists('./data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_0[data_idx][1]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_0[data_idx][1]].replace('/', '_')[:245] + '.pt')
                    continue
                item['mol_0_pt'] = './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_0[data_idx][0]].replace('/', '_')[:245] + '.pt'
                item['mol_1_pt'] = './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_0[data_idx][1]].replace('/', '_')[:245] + '.pt'
                item['label'] = merged_labels_0[data_idx]
                train_data_items_list.append(item)

            for data_idx in range(len(merged_labels_1)):
                item = {}
                item['task_id'] = task_id
                if not os.path.exists('./data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_1[data_idx][0]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_1[data_idx][0]].replace('/', '_')[:245] + '.pt')
                    continue
                if not os.path.exists('./data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_1[data_idx][1]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_1[data_idx][1]].replace('/', '_')[:245] + '.pt')
                    continue
                item['mol_0_pt'] = './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_1[data_idx][0]].replace('/', '_')[:245] + '.pt'
                item['mol_1_pt'] = './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_1[data_idx][1]].replace('/', '_')[:245] + '.pt'
                item['label'] = merged_labels_1[data_idx]
                valid_data_items_list.append(item)

            for data_idx in range(len(merged_labels_2)):
                item = {}
                item['task_id'] = task_id
                if not os.path.exists('./data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_2[data_idx][0]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_2[data_idx][0]].replace('/', '_')[:245] + '.pt')
                    continue
                if not os.path.exists('./data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_2[data_idx][1]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_2[data_idx][1]].replace('/', '_')[:245] + '.pt')
                    continue
                item['mol_0_pt'] = './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_2[data_idx][0]].replace('/', '_')[:245] + '.pt'
                item['mol_1_pt'] = './data_store/DDI_mol_pt/mol_' + ddi_smiles_dict[merged_drug_pairs_2[data_idx][1]].replace('/', '_')[:245] + '.pt'
                item['label'] = merged_labels_2[data_idx]
                test_data_items_list.append(item)
                
            train_data_items_pro, valid_data_items_pro, test_data_items_pro = train_data_items_list, valid_data_items_list, test_data_items_list


        # RYP TDC
        if task_id in ["buchwald-hartwig"]:
        
            if task_id == "buchwald-hartwig":
                data = Yields(name = 'Buchwald-Hartwig', path="./data/RYP_data_from_TDC")
                split = data.get_split()

            train_reaction_list = split['train'].iloc[:, 1].tolist()
            valid_reaction_list = split['valid'].iloc[:, 1].tolist()
            test_reaction_list = split['test'].iloc[:, 1].tolist()
            
            train_labels_list = split['train'].iloc[:, 2].tolist()
            valid_labels_list = split['valid'].iloc[:, 2].tolist()
            test_labels_list = split['test'].iloc[:, 2].tolist()
            
            train_data_items_list = []
            error_count = 0
            
            for idx in range(len(train_reaction_list)):
            
                if ((train_reaction_list[idx]['catalyst'] != '') and (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['catalyst']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['catalyst']].replace('/', '_')[-122:]}.pt"))):
                    print("error catalyst path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['catalyst']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['catalyst']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue

                if (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt")):
                    print("error product path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue
                
                if (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt")):
                    print("error reactant path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue
                
                item = {}
                item['mol_0_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt"
                item['mol_1_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt"
                if train_reaction_list[idx]['catalyst'] == '':
                    item['mol_2_pt'] = None
                else:
                    item['mol_2_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[train_reaction_list[idx]['catalyst']].replace('/', '_')[:122] + ryp_smiles_dict[train_reaction_list[idx]['catalyst']].replace('/', '_')[-122:]}.pt"
                item['label'] = train_labels_list[idx]
                item['task_id'] = task_id
                train_data_items_list.append(item)

            valid_data_items_list = []
            for idx in range(len(valid_reaction_list)):
        
                if ((valid_reaction_list[idx]['catalyst'] != '') and (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['catalyst']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['catalyst']].replace('/', '_')[-122:]}.pt"))):
                    print("error catalyst path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['catalyst']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['catalyst']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue

                if (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt")):
                    print("error product path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue
                
                if (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt")):
                    print("error reactant path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue
                
                item = {}
                item['mol_0_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt"
                item['mol_1_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt"
                if valid_reaction_list[idx]['catalyst'] == '':
                    item['mol_2_pt'] = None
                else:
                    item['mol_2_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[valid_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[valid_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt"
                item['label'] = valid_labels_list[idx]
                item['task_id'] = task_id
                valid_data_items_list.append(item)
            
            test_data_items_list = []
            for idx in range(len(test_reaction_list)):
        
                if ((test_reaction_list[idx]['catalyst'] != '') and (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['catalyst']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['catalyst']].replace('/', '_')[-122:]}.pt"))):
                    print("error catalyst path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['catalyst']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['catalyst']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue

                if (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt")):
                    print("error product path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue
                
                if (not os.path.exists(f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt")):
                    print("error reactant path: ", f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt")
                    error_count += 1
                    continue
                
                item = {}
                item['mol_0_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['product']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['product']].replace('/', '_')[-122:]}.pt"
                item['mol_1_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['reactant']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['reactant']].replace('/', '_')[-122:]}.pt"
                if test_reaction_list[idx]['catalyst'] == '':
                    item['mol_2_pt'] = None
                else:
                    item['mol_2_pt'] = f"./data_store/RYP_mol_pt/mol_{ryp_smiles_dict[test_reaction_list[idx]['catalyst']].replace('/', '_')[:122] + ryp_smiles_dict[test_reaction_list[idx]['catalyst']].replace('/', '_')[-122:]}.pt"
                item['label'] = test_labels_list[idx]
                item['task_id'] = task_id
                test_data_items_list.append(item)

            train_data_items_pro, valid_data_items_pro, test_data_items_pro = train_data_items_list, valid_data_items_list, test_data_items_list

            
        # DRP TDC
        if task_id in ["gdsc1", "gdsc2"]:
            if task_id == "gdsc1":
                DRP_name = 'GDSC1'
            if task_id == "gdsc2":
                DRP_name = 'GDSC2'
            from tdc.multi_pred import DrugRes
            data = DrugRes(name = DRP_name, path="./data/DRP_data_from_TDC")

            split = data.get_split()
            train_cellline_list = split['train'].loc[:, 'Cell Line'].tolist()
            valid_cellline_list = split['valid'].loc[:, 'Cell Line'].tolist()
            test_cellline_list = split['test'].loc[:, 'Cell Line'].tolist()
            train_mol_list = split['train'].loc[:, 'Drug'].tolist()
            valid_mol_list = split['valid'].loc[:, 'Drug'].tolist()
            test_mol_list = split['test'].loc[:, 'Drug'].tolist()
            train_label_list = split['train'].loc[:, 'Y'].tolist()
            valid_label_list = split['valid'].loc[:, 'Y'].tolist()
            test_label_list = split['test'].loc[:, 'Y'].tolist()

            train_data_items_list = []
            valid_data_items_list = []
            test_data_items_list = []
        
            for data_idx in range(len(train_label_list)):
                item = {}
                item['task_id'] = task_id
                try:
                    if not os.path.exists('./data_store/DRP_mol_pt/mol_' + drp_smiles_dict[train_mol_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                        print("error path: ", './data_store/DRP_mol_pt/mol_' + drp_smiles_dict[train_mol_list[data_idx]].replace('/', '_')[:245] + '.pt')
                        continue
                except:
                    continue
                item['mol_pt'] = './data_store/DRP_mol_pt/mol_' + drp_smiles_dict[train_mol_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['cellline'] = train_cellline_list[data_idx]
                item['label'] = train_label_list[data_idx]
                train_data_items_list.append(item)

            for data_idx in range(len(valid_label_list)):
                item = {}
                item['task_id'] = task_id
                try:
                    if not os.path.exists('./data_store/DRP_mol_pt/mol_' + drp_smiles_dict[valid_mol_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                        print("error path: ", './data_store/DRP_mol_pt/mol_' + drp_smiles_dict[valid_mol_list[data_idx]].replace('/', '_')[:245] + '.pt')
                        continue
                except:
                    continue
                item['mol_pt'] = './data_store/DRP_mol_pt/mol_' + drp_smiles_dict[valid_mol_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['cellline'] = valid_cellline_list[data_idx]
                item['label'] = valid_label_list[data_idx]
                valid_data_items_list.append(item)

            for data_idx in range(len(test_label_list)):
                item = {}
                item['task_id'] = task_id
                try:
                    if not os.path.exists('./data_store/DRP_mol_pt/mol_' + drp_smiles_dict[test_mol_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                        print("error path: ", './data_store/DRP_mol_pt/mol_' + drp_smiles_dict[test_mol_list[data_idx]].replace('/', '_')[:245] + '.pt')
                        continue
                except:
                    continue
                item['mol_pt'] = './data_store/DRP_mol_pt/mol_' + drp_smiles_dict[test_mol_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['cellline'] = test_cellline_list[data_idx]
                item['label'] = test_label_list[data_idx]
                test_data_items_list.append(item)

            train_data_items_pro, valid_data_items_pro, test_data_items_pro = train_data_items_list, valid_data_items_list, test_data_items_list

                
        # DSP TDC
        if task_id in ["oncopolypharmacology"]:
            if task_id == "oncopolypharmacology":
                DSP_name = 'OncoPolyPharmacology'
            from tdc.multi_pred import DrugSyn
            data = DrugSyn(name = DSP_name, path="./data/DSP_data_from_TDC")

            split = data.get_split(method='combination')
            train_cellline_list = split['train'].loc[:, 'Cell_Line'].tolist()
            valid_cellline_list = split['valid'].loc[:, 'Cell_Line'].tolist()
            test_cellline_list = split['test'].loc[:, 'Cell_Line'].tolist()
            train_celllineid_list = split['train'].loc[:, 'Cell_Line_ID'].tolist()
            valid_celllineid_list = split['valid'].loc[:, 'Cell_Line_ID'].tolist()
            test_celllineid_list = split['test'].loc[:, 'Cell_Line_ID'].tolist()
            train_mol_0_list = split['train'].loc[:, 'Drug1'].tolist()
            valid_mol_0_list = split['valid'].loc[:, 'Drug1'].tolist()
            test_mol_0_list = split['test'].loc[:, 'Drug1'].tolist()
            train_mol_1_list = split['train'].loc[:, 'Drug2'].tolist()
            valid_mol_1_list = split['valid'].loc[:, 'Drug2'].tolist()
            test_mol_1_list = split['test'].loc[:, 'Drug2'].tolist()
            train_label_list = split['train'].loc[:, 'Y'].tolist()
            valid_label_list = split['valid'].loc[:, 'Y'].tolist()
            test_label_list = split['test'].loc[:, 'Y'].tolist()

            train_data_items_list = []
            valid_data_items_list = []
            test_data_items_list = []
        
            for data_idx in range(len(train_label_list)):
                item = {}
                item['task_id'] = task_id
                if not os.path.exists('./data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[train_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[train_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt')
                    continue
                if not os.path.exists('./data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[train_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[train_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt')
                    continue
                item['mol_0_pt'] = './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[train_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['mol_1_pt'] = './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[train_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['cellline'] = train_cellline_list[data_idx]
                item['label'] = train_label_list[data_idx]
                train_data_items_list.append(item)

            for data_idx in range(len(valid_label_list)):
                item = {}
                item['task_id'] = task_id
                if not os.path.exists('./data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[valid_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[valid_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt')
                    continue
                if not os.path.exists('./data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[valid_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[valid_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt')
                    continue
                item['mol_0_pt'] = './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[valid_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['mol_1_pt'] = './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[valid_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['cellline'] = valid_cellline_list[data_idx]
                item['label'] = valid_label_list[data_idx]
                valid_data_items_list.append(item)

            for data_idx in range(len(test_label_list)):
                item = {}
                item['task_id'] = task_id
                if not os.path.exists('./data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[test_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[test_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt')
                    continue
                if not os.path.exists('./data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[test_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt'):
                    print("error path: ", './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[test_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt')
                    continue
                item['mol_0_pt'] = './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[test_mol_0_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['mol_1_pt'] = './data_store/DSP_mol_pt/mol_' + dsp_smiles_dict[test_mol_1_list[data_idx]].replace('/', '_')[:245] + '.pt'
                item['cellline'] = test_cellline_list[data_idx]
                item['label'] = test_label_list[data_idx]
                test_data_items_list.append(item)

            train_data_items_pro, valid_data_items_pro, test_data_items_pro = train_data_items_list, valid_data_items_list, test_data_items_list
            
        
        if task_id in ["pgp_broccatelli"]:
            weight = 126
        if task_id in ["bioavailability_ma"]:
            weight = 33
        if task_id in ["bbb_martins", "cyp3a4_substrate_carbonmangels", "cyp2d6_veith", "cyp3a4_veith", "cyp2d6_substrate_carbonmangels", "esol", "lipophilicity", "bbbp", "sider", "hiv"]:
            weight = 3
        if task_id in ["cyp2c9_substrate_carbonmangels"]:
            weight = 5
        if task_id in ["ames", "cyp2c9_veith", "solubility_aqsoldb"]:
            weight = 6
        if task_id in ["caco2_wang"]:
            weight = 113
        if task_id in ["lipophilicity_astrazeneca", "ppbr_az", "ld50_zhu", "vdss_lombardo", "dili", "herg", "hia_hou"]:
            weight = 11
        if task_id in ["ld50_zhu", "clearance_microsome_az"]:
            weight = 20
        if task_id in ["half_life_obach"]:
            weight = 76
        if task_id in ["clearance_hepatocyte_az", "clintox"]:
            weight = 61
        if task_id in ["davis", "kiba", "drugbank", "gdsc1", "gdsc2"]:
            weight = 1
        if task_id in ["bindingdb_kd"]:
            weight = 2
        if task_id in ["oncopolypharmacology"]:
            weight = 8
        if task_id in ["buchwald-hartwig"]:
            weight = 32
        print(task_id, weight)
        for weight_i in range(weight):
            train_dataset_list += train_data_items_pro
        valid_dataset_list += valid_data_items_pro
        test_dataset_list += test_data_items_pro

        print("train_data_items_pro: ", len(train_data_items_pro))
        print("valid_data_items_pro: ", len(valid_data_items_pro))
        print("test_data_items_pro: ", len(test_data_items_pro))
        
    return train_dataset_list, valid_dataset_list, test_dataset_list

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
    
class MoleculeDataset(Dataset):

    @staticmethod
    def calculate_descriptors(mol):
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

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """Get Morgan fingerprint."""
        nBits = 200
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """Get MACCS fingerprint."""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    @staticmethod
    def get_daylight_functional_group_counts(mol, fg_mo_list):
        """Get Daylight functional group counts."""
        fg_counts = []
        for fg_mol in fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def split_descriptor(descriptor):
        bool_list = torch.tensor([False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True])
        return torch.tensor(descriptor, dtype=torch.float)[bool_list], torch.tensor(descriptor, dtype=torch.float)[~bool_list]

    def __init__(self, dataset_list, model_type):
        super(Dataset, self).__init__()
#        self.task_type = task_type
        self.data_items = dataset_list
        self.model_type = model_type
        with open('desc_int_config.json', 'r', encoding='utf-8') as f:
            self.int_list = json.load(f)
        self.DTI_mol_pt_file_dict = {}
        self.DTI_prot_pt_file_dict = {}
        self.DDI_mol_0_pt_file_dict = {}
        self.DDI_mol_1_pt_file_dict = {}
        self.DRP_mol_pt_file_dict = {}
        self.DRP_cellline_pt_file_dict = {}
        self.DSP_mol_0_pt_file_dict = {}
        self.DSP_mol_1_pt_file_dict = {}
        self.DSP_cellline_pt_file_dict = {}
        self.RYP_mol_0_pt_file_dict = {}
        self.RYP_mol_1_pt_file_dict = {}
        self.RYP_mol_2_pt_file_dict = {}
        dataset_list_len = len(dataset_list)
        print("dataset_list: ", dataset_list_len)
        item_idx = 0
        for item in dataset_list:
            if item_idx % 1000 == 0:
                print(item_idx)
            item_idx += 1
            if item['task_id'] in ['davis', 'kiba', 'bindingdb_kd']:
                if item['mol_pt'] not in self.DTI_mol_pt_file_dict.keys():
                    self.DTI_mol_pt_file_dict[item['mol_pt']] = self.get_item_dp(torch.load(item['mol_pt']), model_type)
                if item['protein_rep'] not in self.DTI_prot_pt_file_dict.keys():
                    self.DTI_prot_pt_file_dict[item['protein_rep']] = torch.load(item['protein_rep'])
            if item['task_id'] in ["drugbank", "twosides"]:
                if item['mol_0_pt'] not in self.DDI_mol_0_pt_file_dict.keys():
                    self.DDI_mol_0_pt_file_dict[item['mol_0_pt']] = self.get_item_dp(torch.load(item['mol_0_pt']), model_type)
                if item['mol_1_pt'] not in self.DDI_mol_1_pt_file_dict.keys():
                    self.DDI_mol_1_pt_file_dict[item['mol_1_pt']] = self.get_item_dp(torch.load(item['mol_1_pt']), model_type)
            if item['task_id'] in ["buchwald-hartwig"]:
                if item['mol_0_pt'] not in self.RYP_mol_0_pt_file_dict.keys():
                    self.RYP_mol_0_pt_file_dict[item['mol_0_pt']] = self.get_item_dp(torch.load(item['mol_0_pt']), model_type)
                if item['mol_1_pt'] not in self.RYP_mol_1_pt_file_dict.keys():
                    self.RYP_mol_1_pt_file_dict[item['mol_1_pt']] = self.get_item_dp(torch.load(item['mol_1_pt']), model_type)
                if (item['mol_2_pt'] not in self.RYP_mol_2_pt_file_dict.keys()) and item['mol_2_pt'] is not None:
                    self.RYP_mol_2_pt_file_dict[item['mol_2_pt']] = self.get_item_dp(torch.load(item['mol_2_pt']), model_type)
            if item['task_id'] in ["gdsc1", "gdsc2"]:
                if item['mol_pt'] not in self.DRP_mol_pt_file_dict.keys():
                    self.DRP_mol_pt_file_dict[item['mol_pt']] = self.get_item_dp(torch.load(item['mol_pt']), model_type)
                if str(item['cellline']) not in self.DRP_cellline_pt_file_dict.keys():
                    self.DRP_cellline_pt_file_dict[str(item['cellline'])] = torch.tensor(item['cellline'], dtype=torch.float32)
            if item['task_id'] in ["oncopolypharmacology"]:
                if item['mol_0_pt'] not in self.DSP_mol_0_pt_file_dict.keys():
                    self.DSP_mol_0_pt_file_dict[item['mol_0_pt']] = self.get_item_dp(torch.load(item['mol_0_pt']), model_type)
                if item['mol_1_pt'] not in self.DSP_mol_1_pt_file_dict.keys():
                    self.DSP_mol_1_pt_file_dict[item['mol_1_pt']] = self.get_item_dp(torch.load(item['mol_1_pt']), model_type)
                if str(item['cellline']) not in self.DSP_cellline_pt_file_dict.keys():
                    self.DSP_cellline_pt_file_dict[str(item['cellline'])] = torch.tensor(item['cellline'], dtype=torch.float32)
            
        self.default_c = torch.load("catalyst_default.pt")
        self.default_c["fraginfo"] = Mol2HeteroGraphData(Chem.MolFromSmiles(self.default_c['smiles']))
        self.default_c["fraggraph"] = Data2HeteroGraph(self.default_c["fraginfo"])
        self.default_c = self.get_item_dp(self.default_c, self.model_type)

        
    @staticmethod
    def get_item_dp(data_item, model_type):
        if True:
            data_item['descriptor_float'] = data_item['descriptor_float'].clamp(max=100000)
            
            for i in range(len(data_item['descriptor_int'])):
                data_item['descriptor_int'][i] = data_item['descriptor_int'][i].clamp(min=int_list[i][1], max=int_list[i][0])
            
            data_item['daylight_data'] = data_item['daylight_data'].clamp(min=0, max=100)

            data_item['mask_3d'] = pad_node_features(data_item['pos'], 200)
            
            data_item['x'] = torch.nan_to_num(data_item['x'], nan=0, posinf=0, neginf=0).to(torch.int64)
            data_item['edge_index'] = torch.nan_to_num(data_item['edge_index'], nan=0, posinf=0, neginf=0).to(torch.int64)
            data_item['edge_attr'] = torch.nan_to_num(data_item['edge_attr'], nan=0, posinf=0, neginf=0).to(torch.int64)
            data_item['pos'] = torch.nan_to_num(data_item['pos'], nan=0, posinf=0, neginf=0).to(torch.float32)
            data_item['bond_lengths'] = torch.nan_to_num(data_item['bond_lengths'], nan=0, posinf=0, neginf=0).to(torch.float32)
            data_item['super_edges'] = torch.nan_to_num(data_item['super_edges'], nan=0, posinf=0, neginf=0).to(torch.int64)
            data_item['bond_angles'] = torch.nan_to_num(data_item['bond_angles'], nan=0, posinf=0, neginf=0).to(torch.float32)
            data_item['morgan_data'] = torch.nan_to_num(data_item['morgan_data'], nan=0, posinf=0, neginf=0).to(torch.float32)
            data_item['maccs_data'] = torch.nan_to_num(data_item['maccs_data'], nan=0, posinf=0, neginf=0).to(torch.float32)
            data_item['daylight_data'] = torch.nan_to_num(data_item['daylight_data'], nan=0, posinf=0, neginf=0).to(torch.float32)
            data_item['descriptor_int'] = torch.nan_to_num(data_item['descriptor_int'], nan=0, posinf=0, neginf=0).to(torch.float32)
            data_item['descriptor_float'] = torch.nan_to_num(data_item['descriptor_float'], nan=0, posinf=0, neginf=0).to(torch.float32)
            data_item['src_mask'] = torch.nan_to_num(data_item['src_mask'], nan=0, posinf=0, neginf=0).to(torch.bool)
            data_item['mask_3d'] = torch.nan_to_num(data_item['mask_3d'], nan=0, posinf=0, neginf=0).to(torch.bool)
            edge_attr = torch.cat([data_item['edge_attr'][:-len(data_item['x'])] + 1, data_item['edge_attr'][-len(data_item['x']):]], dim=0)


            data_2D = data_item["fraggraph"]

            data_3D1 = Data(x=data_item['x'], pos=data_item['pos'], edge_index=data_item['edge_index'], edge_attr=edge_attr, bond_lengths=data_item['bond_lengths'])
        
            data_3D2 = Data(x=edge_attr, edge_index=data_item['super_edges'], edge_attr=data_item['bond_angles'])

            return data_2D, data_3D1, data_3D2, data_item
        
    def __getitem__(self, index):

        
        try:
            data_item = self.data_items[index]
            if data_item['task_id'] in ['ames', 'bbb_martins', 'bioavailability_ma', 'cyp2c9_substrate_carbonmangels', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith', 'cyp3a4_substrate_carbonmangels', 'cyp3a4_veith', 'dili', 'herg', 'hia_hou', 'pgp_broccatelli', 'caco2_wang', 'clearance_hepatocyte_az', 'clearance_microsome_az', 'lipophilicity', 'half_life_obach', 'ld50_zhu', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb', 'vdss_lombardo', 'bbbp', 'clintox', 'sider', 'hiv', 'esol']:
                if len(self.data_items[index]['pos']) > 200:
                    next_index = (index + random.randrange(len(self))) % len(self)
                    return self.__getitem__(next_index)
                data_2D, data_3D1, data_3D2, data_item_tmp = self.get_item_dp(data_item, self.model_type)

                return data_2D, data_3D1, data_3D2, data_item_tmp['morgan_data'].unsqueeze(dim=-1), data_item_tmp['maccs_data'].unsqueeze(dim=-1), data_item_tmp['daylight_data'].unsqueeze(dim=-1), data_item_tmp['descriptor_int'].unsqueeze(dim=-1), data_item_tmp['descriptor_float'].unsqueeze(dim=-1), data_item_tmp['src_mask'], data_item_tmp['mask_3d'], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, data_item_tmp['label'].reshape(-1), data_item_tmp['task_id']
                
            if data_item['task_id'] in ['davis', 'kiba', 'bindingdb_kd']:
                labels = torch.tensor(data_item['label'])
                prot_rep = self.DTI_prot_pt_file_dict[data_item['protein_rep']]
                data_2D, data_3D1, data_3D2, data_item_mol = self.DTI_mol_pt_file_dict[data_item['mol_pt']]
                prot_rep = torch.nan_to_num(prot_rep, nan=0, posinf=0, neginf=0)
                
                return data_2D, data_3D1, data_3D2, data_item_mol['morgan_data'].unsqueeze(dim=-1), data_item_mol['maccs_data'].unsqueeze(dim=-1), data_item_mol['daylight_data'].unsqueeze(dim=-1), data_item_mol['descriptor_int'].unsqueeze(dim=-1), data_item_mol['descriptor_float'].unsqueeze(dim=-1), data_item_mol['src_mask'], data_item_mol['mask_3d'], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, prot_rep, None, None, labels.reshape(-1), data_item['task_id']

            if data_item['task_id'] in ["drugbank", "twosides"]:

                labels = torch.tensor(data_item['label'])
                data_2D_0, data_3D1_0, data_3D2_0, mol_0 = self.DDI_mol_0_pt_file_dict[data_item['mol_0_pt']]
                data_2D_1, data_3D1_1, data_3D2_1, mol_1 = self.DDI_mol_1_pt_file_dict[data_item['mol_1_pt']]

                return data_2D_0, data_3D1_0, data_3D2_0, mol_0['morgan_data'].unsqueeze(dim=-1), mol_0['maccs_data'].unsqueeze(dim=-1), mol_0['daylight_data'].unsqueeze(dim=-1), mol_0['descriptor_int'].unsqueeze(dim=-1), mol_0['descriptor_float'].unsqueeze(dim=-1), mol_0['src_mask'], mol_0['mask_3d'], data_2D_1, data_3D1_1, data_3D2_1, mol_1['morgan_data'].unsqueeze(dim=-1), mol_1['maccs_data'].unsqueeze(dim=-1), mol_1['daylight_data'].unsqueeze(dim=-1), mol_1['descriptor_int'].unsqueeze(dim=-1), mol_1['descriptor_float'].unsqueeze(dim=-1), mol_1['src_mask'], mol_1['mask_3d'], None, None, None, None, None, None, None, None, None, None, None, None, None, labels.reshape(-1), data_item['task_id']

            if data_item['task_id'] in ["buchwald-hartwig"]:
                labels = torch.tensor(data_item['label'])
                data_2D_0, data_3D1_0, data_3D2_0, mol_0 = self.RYP_mol_0_pt_file_dict[data_item['mol_0_pt']]
                data_2D_1, data_3D1_1, data_3D2_1, mol_1 = self.RYP_mol_1_pt_file_dict[data_item['mol_1_pt']]
                if data_item['mol_2_pt'] is not None:
                    data_2D_2, data_3D1_2, data_3D2_2, mol_2 = self.RYP_mol_2_pt_file_dict[data_item['mol_2_pt']]
                    data_item['is_mol_2'] = True
                else:
                    data_2D_2, data_3D1_2, data_3D2_2, mol_2 = self.default_c
                    data_item['is_mol_2'] = False
                
                return data_2D_0, data_3D1_0, data_3D2_0, mol_0['morgan_data'].unsqueeze(dim=-1), mol_0['maccs_data'].unsqueeze(dim=-1), mol_0['daylight_data'].unsqueeze(dim=-1), mol_0['descriptor_int'].unsqueeze(dim=-1), mol_0['descriptor_float'].unsqueeze(dim=-1), mol_0['src_mask'], mol_0['mask_3d'], data_2D_1, data_3D1_1, data_3D2_1, mol_1['morgan_data'].unsqueeze(dim=-1), mol_1['maccs_data'].unsqueeze(dim=-1), mol_1['daylight_data'].unsqueeze(dim=-1), mol_1['descriptor_int'].unsqueeze(dim=-1), mol_1['descriptor_float'].unsqueeze(dim=-1), mol_1['src_mask'], mol_1['mask_3d'], data_2D_2, data_3D1_2, data_3D2_2, mol_2['morgan_data'].unsqueeze(dim=-1), mol_2['maccs_data'].unsqueeze(dim=-1), mol_2['daylight_data'].unsqueeze(dim=-1), mol_2['descriptor_int'].unsqueeze(dim=-1), mol_2['descriptor_float'].unsqueeze(dim=-1), mol_2['src_mask'], mol_2['mask_3d'], None, None, data_item['is_mol_2'], labels.reshape(-1), data_item['task_id']

            if data_item['task_id'] in ["gdsc1", "gdsc2"]:
                labels = torch.tensor(data_item['label'])
                data_2D_0, data_3D1_0, data_3D2_0, mol_0 = self.DRP_mol_pt_file_dict[data_item['mol_pt']]
                cellline = self.DRP_cellline_pt_file_dict[str(data_item['cellline'])]

                return data_2D_0, data_3D1_0, data_3D2_0, mol_0['morgan_data'].unsqueeze(dim=-1), mol_0['maccs_data'].unsqueeze(dim=-1), mol_0['daylight_data'].unsqueeze(dim=-1), mol_0['descriptor_int'].unsqueeze(dim=-1), mol_0['descriptor_float'].unsqueeze(dim=-1), mol_0['src_mask'], mol_0['mask_3d'], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, cellline, None, labels.reshape(-1), data_item['task_id']

            if data_item['task_id'] in ["oncopolypharmacology", "drugcomb"]:
                labels = torch.tensor(data_item['label'])
                data_2D_0, data_3D1_0, data_3D2_0, mol_0 = self.DSP_mol_0_pt_file_dict[data_item['mol_0_pt']]
                data_2D_1, data_3D1_1, data_3D2_1, mol_1 = self.DSP_mol_1_pt_file_dict[data_item['mol_1_pt']]
                cellline = self.DSP_cellline_pt_file_dict[str(data_item['cellline'])]

                return data_2D_0, data_3D1_0, data_3D2_0, mol_0['morgan_data'].unsqueeze(dim=-1), mol_0['maccs_data'].unsqueeze(dim=-1), mol_0['daylight_data'].unsqueeze(dim=-1), mol_0['descriptor_int'].unsqueeze(dim=-1), mol_0['descriptor_float'].unsqueeze(dim=-1), mol_0['src_mask'], mol_0['mask_3d'], data_2D_1, data_3D1_1, data_3D2_1, mol_1['morgan_data'].unsqueeze(dim=-1), mol_1['maccs_data'].unsqueeze(dim=-1), mol_1['daylight_data'].unsqueeze(dim=-1), mol_1['descriptor_int'].unsqueeze(dim=-1), mol_1['descriptor_float'].unsqueeze(dim=-1), mol_1['src_mask'], mol_1['mask_3d'], None, None, None, None, None, None, None, None, None, None, None, cellline, None, labels.reshape(-1), data_item['task_id']
                
        except:
            next_index = (index + random.randrange(len(self))) % len(self)
            return self.__getitem__(next_index) 

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

        prot_list = []
        prot_num_list = []
        cellline_drp_list = []
        cellline_dsp_list = []
        is_catalyst_list = []
        labels_list = []
        tasks_list = []

        for data_item in batch_data_list:

            if data_item[34] in ['ames', 'bbb_martins', 'bioavailability_ma', 'cyp2c9_substrate_carbonmangels', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2d6_veith', 'cyp3a4_substrate_carbonmangels', 'cyp3a4_veith', 'dili', 'herg', 'hia_hou', 'pgp_broccatelli', 'caco2_wang', 'clearance_hepatocyte_az', 'clearance_microsome_az', 'lipophilicity', 'half_life_obach', 'ld50_zhu', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb', 'vdss_lombardo', 'bbbp', 'clintox', 'sider', 'hiv', 'esol']:
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
                
            if data_item[34] in ['davis', 'kiba', 'bindingdb_kd']:
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
                prot_num_list.append(data_item[30].shape[1])
                prot_list.append(data_item[30])
                
            if data_item[34] in ["drugbank", "twosides"]:
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
                data_2D_list.append(data_item[10])
                data_3D1_list.append(data_item[11])
                data_3D2_list.append(data_item[12])
                morgan_list.append(data_item[13].unsqueeze(0))
                maccs_list.append(data_item[14].unsqueeze(0))
                daylight_list.append(data_item[15].unsqueeze(0))
                descriptor_int_list.append(data_item[16].unsqueeze(0))
                descriptor_float_list.append(data_item[17].unsqueeze(0))
                src_mask_list.append(data_item[18].unsqueeze(0))
                mask_3d_list.append(data_item[19].unsqueeze(0))

            if data_item[34] in ["buchwald-hartwig"]:
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
                data_2D_list.append(data_item[10])
                data_3D1_list.append(data_item[11])
                data_3D2_list.append(data_item[12])
                morgan_list.append(data_item[13].unsqueeze(0))
                maccs_list.append(data_item[14].unsqueeze(0))
                daylight_list.append(data_item[15].unsqueeze(0))
                descriptor_int_list.append(data_item[16].unsqueeze(0))
                descriptor_float_list.append(data_item[17].unsqueeze(0))
                src_mask_list.append(data_item[18].unsqueeze(0))
                mask_3d_list.append(data_item[19].unsqueeze(0))
                data_2D_list.append(data_item[20])
                data_3D1_list.append(data_item[21])
                data_3D2_list.append(data_item[22])
                morgan_list.append(data_item[23].unsqueeze(0))
                maccs_list.append(data_item[24].unsqueeze(0))
                daylight_list.append(data_item[25].unsqueeze(0))
                descriptor_int_list.append(data_item[26].unsqueeze(0))
                descriptor_float_list.append(data_item[27].unsqueeze(0))
                src_mask_list.append(data_item[28].unsqueeze(0))
                mask_3d_list.append(data_item[29].unsqueeze(0))
                is_catalyst_list.append(data_item[32])

            if data_item[34] in ["gdsc1", "gdsc2"]:
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
                cellline_drp_list.append(data_item[31].unsqueeze(0))
                
            if data_item[34] in ["oncopolypharmacology", "drugcomb"]:
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
                data_2D_list.append(data_item[10])
                data_3D1_list.append(data_item[11])
                data_3D2_list.append(data_item[12])
                morgan_list.append(data_item[13].unsqueeze(0))
                maccs_list.append(data_item[14].unsqueeze(0))
                daylight_list.append(data_item[15].unsqueeze(0))
                descriptor_int_list.append(data_item[16].unsqueeze(0))
                descriptor_float_list.append(data_item[17].unsqueeze(0))
                src_mask_list.append(data_item[18].unsqueeze(0))
                mask_3d_list.append(data_item[19].unsqueeze(0))
                cellline_dsp_list.append(data_item[31].unsqueeze(0))

            
            labels_list.append(data_item[33])
            tasks_list.append(data_item[34])

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
        if len(prot_list) > 0:
            prot_list = torch.cat(prot_list, dim=1)
        if len(cellline_drp_list) > 0:
            cellline_drp_list = torch.cat(cellline_drp_list, dim=0)
        if len(cellline_dsp_list) > 0:
            cellline_dsp_list = torch.cat(cellline_dsp_list, dim=0)
        labels_list = torch.cat(labels_list)
        
        return data_2D_list, data_3D1_list, data_3D2_list, morgan_list, maccs_list, daylight_list, descriptor_int_list, descriptor_float_list, src_mask_list, mask_3d_list, prot_list, prot_num_list, cellline_drp_list, cellline_dsp_list, is_catalyst_list, tasks_list, labels_list


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy('file_system')
            
class MolTestDatasetWrapper(object):
    def __init__(self, batch_size, num_workers):
        super(object, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        print("MolTestDatasetWrapper init done")

    def get_data_loaders(self, model_type):
        train_dataset_list, valid_dataset_list, test_dataset_list = get_dataset(model_type)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset_list, valid_dataset_list, test_dataset_list, model_type)
        return train_loader, valid_loader, test_loader
        

    def get_train_validation_data_loaders(self, train_dataset_list, valid_dataset_list, test_dataset_list, model_type):

        train_dataset, valid_dataset, test_dataset = MoleculeDataset(train_dataset_list, model_type), MoleculeDataset(valid_dataset_list, model_type), MoleculeDataset(test_dataset_list, model_type)
        del train_dataset_list
        del valid_dataset_list
        del test_dataset_list

        print("train_dataset: ", len(train_dataset))
        print("valid_dataset: ", len(valid_dataset))
        print("test_dataset: ", len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=1, drop_last=False, shuffle=True, collate_fn=GeoPredCollateFn(model_type), persistent_workers=True, pin_memory=True, timeout=0, worker_init_fn=set_worker_sharing_strategy)

        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=1, drop_last=False, shuffle=True, collate_fn=GeoPredCollateFn(model_type), persistent_workers=True, pin_memory=True, timeout=0, worker_init_fn=set_worker_sharing_strategy)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=1, drop_last=False, shuffle=True, collate_fn=GeoPredCollateFn(model_type), persistent_workers=True, pin_memory=True, timeout=0, worker_init_fn=set_worker_sharing_strategy)


        del train_dataset
        del valid_dataset
        del test_dataset

        return train_loader, valid_loader, test_loader


        