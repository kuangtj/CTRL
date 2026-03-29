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

def get_dataset(task_id_want):
    import json

    file_path = f'./data/DRP_data_from_TDC/{task_id_want}_data.json'

    with open(file_path, 'r', encoding='utf-8') as file:
        smiles_list = json.load(file)
       
    signal.signal(signal.SIGALRM, timeout_handler)
    for idx in range(len(smiles_list)):
        try:
            signal.alarm(600)
            mol_tmp = Chem.MolFromSmiles(smiles_list[idx])
            if mol_tmp.GetNumHeavyAtoms() > 200:
                print(f"The mol is too large(>200): {smiles_list[idx]}")
            mol_smiles = Chem.MolToSmiles(mol_tmp)
            data_item = smiles_preporcess(mol_smiles, f"./data_store/DRP_mol_sdf/mol_{mol_smiles.replace('/', '_')[:245]}.sdf")
            torch.save(data_item, f"./data_store/DRP_mol_pt/mol_{mol_smiles.replace('/', '_')[:245]}.pt")
                        
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
    parser.add_argument('--task_id', type=str)
    args = parser.parse_args()
    get_dataset(args.task_id)
