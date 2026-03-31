import pandas as pd
import os

from data_process import smiles_preporcess
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from tdc.single_pred import ADME
from tdc.single_pred import Tox
import signal 

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout !!!")

def get_smiles_list(file_path):
    smiles_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            clean_smiles = line.strip()
            if clean_smiles: 
                smiles_list.append(clean_smiles)
                
    return smiles_list
    
def get_dataset(dataset_name, chunk_idx):
    if dataset_name == "pubchem":
        file_path = "./data/pretrain_data/pubchem-10m-clean.txt"
    if dataset_name == "zinc":
        file_path = "./data/pretrain_data/zinc_full.txt"
        
    smiles_list_all = get_smiles_list(file_path)

    chunk_len = int(len(smiles_list_all)/20) + 1

    smiles_list = smiles_list_all[chunk_idx * chunk_len: (chunk_idx + 1) * chunk_len]
    
    signal.signal(signal.SIGALRM, timeout_handler)

    for idx in range(len(smiles_list)):
        try:
            signal.alarm(600)
            mol_tmp = Chem.MolFromSmiles(smiles_list[idx])
            if mol_tmp.GetNumHeavyAtoms() > 200:
                print(f"The mol is too large(>200): {smiles_list[idx]}")
                continue
            mol_smiles = Chem.MolToSmiles(mol_tmp)
            data_sdf = f"./data_store/pretrain_data_sdf/{dataset_name}_{mol_smiles.replace('/', '_')[:240]}.sdf"
            data_pt = f"./data_store/pretrain_data_pt/{dataset_name}_{mol_smiles.replace('/', '_')[:240]}.pt"
            data_item = smiles_preporcess(mol_smiles, data_sdf, pretrain=True)
            torch.save(data_item, data_pt)
                        
        except TimeoutException:
            print(f"Timeout : {smiles_list[idx]}")
            continue
                        
        except Exception as e:
            print(f"Error: {smiles_list[idx]} | {e}")
            continue
                        
        finally:
            signal.alarm(0)
    
    print("save mol_data_list")
    
    
import argparse
import torch.multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--idx', type=int)
    args = parser.parse_args()
    get_dataset(args.dataset, args.idx)






