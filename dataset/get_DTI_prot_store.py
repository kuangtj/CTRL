import pandas as pd
import os

from data_process import prot_preporcess
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

def get_dataset(task_id_want, chunk_idx, device):
    import json

    file_path = f'./data/DTI_data_from_TDC/{task_id_want}_prot_data.json'

    with open(file_path, 'r', encoding='utf-8') as file:
        prot_list = json.load(file)
    
    for idx in range(len(prot_list)):
        try:
            prot_name = prot_list[idx][0]
            prot_seq = prot_list[idx][1]
            print("prot_name: ", prot_name)
            print("prot_seq: ", prot_seq)
            data_item = prot_preporcess(prot_seq, device)
            torch.save(data_item, f"./data_store/DTI_prot_pt/prot_{prot_name.replace('/', '_')[:245]}.pt")
            print(f"./data_store/DTI_prot_pt/prot_{prot_name.replace('/', '_')[:245]}.pt")
        except:
            print("error: ", prot_name)
    
    print("save mol_data_list")
    
    
import argparse
import torch.multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=str)
    parser.add_argument('--idx', type=int)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()
    get_dataset(args.task_id, args.idx, args.device)


    
