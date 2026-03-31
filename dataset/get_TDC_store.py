import pandas as pd
import os

datasets = [
    './data/ADMET_data_from_TDC/ames.tab',
    './data/ADMET_data_from_TDC/bbb_martins.tab',
    './data/ADMET_data_from_TDC/bioavailability_ma.tab',
    './data/ADMET_data_from_TDC/caco2_wang.tab',
    './data/ADMET_data_from_TDC/clearance_hepatocyte_az.tab',
    './data/ADMET_data_from_TDC/clearance_microsome_az.tab',
    './data/ADMET_data_from_TDC/dili.tab',
    './data/ADMET_data_from_TDC/half_life_obach.tab',
    './data/ADMET_data_from_TDC/herg.tab',
    './data/ADMET_data_from_TDC/hia_hou.tab',
    './data/ADMET_data_from_TDC/ld50_zhu.tab',
    './data/ADMET_data_from_TDC/lipophilicity_astrazeneca.tab',
    './data/ADMET_data_from_TDC/pgp_broccatelli.tab',
    './data/ADMET_data_from_TDC/ppbr_az.tab',
    './data/ADMET_data_from_TDC/solubility_aqsoldb.tab',
    './data/ADMET_data_from_TDC/vdss_lombardo.tab',
]

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
    if task_id_want == 'ames':
        data = Tox(name = 'AMES', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'bbb_martins':
        data = ADME(name = 'BBB_Martins', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'bioavailability_ma':
        data = ADME(name = 'Bioavailability_Ma', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'caco2_wang':
        data = ADME(name = 'Caco2_Wang', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'clearance_hepatocyte_az':
        data = ADME(name = 'Clearance_Hepatocyte_AZ', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'clearance_microsome_az':
        data = ADME(name = 'Clearance_Microsome_AZ', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'dili':
        data = Tox(name = 'DILI', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'half_life_obach':
        data = ADME(name = 'Half_Life_Obach', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'herg':
        data = Tox(name = 'hERG', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'hia_hou':
        data = ADME(name = 'HIA_Hou', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'ld50_zhu':
        data = Tox(name = 'LD50_Zhu', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'lipophilicity_astrazeneca':
        data = ADME(name = 'Lipophilicity_AstraZeneca', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'pgp_broccatelli':
        data = ADME(name = 'Pgp_Broccatelli', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'ppbr_az':
        data = ADME(name = 'PPBR_AZ', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'solubility_aqsoldb':
        data = ADME(name = 'Solubility_AqSolDB', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'vdss_lombardo':
        data = ADME(name = 'VDss_Lombardo', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'cyp2d6_veith':
        data = ADME(name = 'CYP2D6_Veith', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'cyp3a4_veith':
        data = ADME(name = 'CYP3A4_Veith', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'cyp2c9_veith':
        data = ADME(name = 'CYP2C9_Veith', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'cyp3a4_substrate_carbonmangels':
        data = ADME(name = 'CYP3A4_Substrate_CarbonMangels', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'cyp2d6_substrate_carbonmangels':
        data = ADME(name = 'CYP2D6_Substrate_CarbonMangels', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want == 'cyp2c9_substrate_carbonmangels':
        data = ADME(name = 'CYP2C9_Substrate_CarbonMangels', path='./data/ADMET_data_from_TDC')
        split = data.get_split(method='scaffold', seed=2026, frac=[0.7, 0.1, 0.2])
    if task_id_want in ['ames', 'bbb_martins', 'bioavailability_ma', 'caco2_wang', 'dili', 'herg', 'hia_hou', 'lipophilicity_astrazeneca', 'pgp_broccatelli', 'ppbr_az', 'solubility_aqsoldb']:
        smiles_name = 'Drug'
    else:
        smiles_name = 'Drug'
        
    train_smiles_list = split['train'][smiles_name].tolist()
    train_label_list = split['train']['Y'].tolist()
    train_data_list = []
    task_id = task_id_want
    
    signal.signal(signal.SIGALRM, timeout_handler)
    for idx in range(len(train_smiles_list)):
        try:
            signal.alarm(600)
            mol_tmp = Chem.MolFromSmiles(train_smiles_list[idx])
            if mol_tmp.GetNumHeavyAtoms() > 200:
                print(f"The mol is too large(>200): {train_smiles_list[idx]}")
                continue
            mol_smiles = Chem.MolToSmiles(mol_tmp)
            data_item = smiles_preporcess(mol_smiles, f"./data_store/{task_id}_mol_sdf/mol_{mol_smiles.replace('/', '_')[:245]}.sdf")
            data_item['label'] = torch.tensor(train_label_list[idx])
            data_item['task_id'] = task_id
            train_data_list.append(data_item)
                        
        except TimeoutException:
            print(f"Timeout : {train_smiles_list[idx]}")
            continue
                        
        except Exception as e:
            # 捕获其他的正常报错
            print(f"Error: {train_smiles_list[idx]} | {e}")
            continue
                        
        finally:
            signal.alarm(0)
    torch.save(train_data_list, f"./data_store/train_{task_id}.pt")
    print("save train_data_list")
    
    valid_smiles_list = split['valid'][smiles_name].tolist()
    valid_label_list = split['valid']['Y'].tolist()
    valid_data_list = []
    for idx in range(len(valid_smiles_list)):
        try:
            signal.alarm(600)
            mol_tmp = Chem.MolFromSmiles(valid_smiles_list[idx])
            if mol_tmp.GetNumHeavyAtoms() > 200:
                print(f"The mol is too large(>200): {valid_smiles_list[idx]}")
                continue
            mol_smiles = Chem.MolToSmiles(mol_tmp)
            data_item = smiles_preporcess(mol_smiles, f"./data_store/{task_id}_mol_sdf/mol_{mol_smiles.replace('/', '_')[:245]}.sdf")
            data_item['label'] = torch.tensor(valid_label_list[idx])
            data_item['task_id'] = task_id
            valid_data_list.append(data_item)
                        
        except TimeoutException:
            print(f"Timeout : {valid_smiles_list[idx]}")
            continue
                        
        except Exception as e:
            print(f"Error: {valid_smiles_list[idx]} | {e}")
            continue
                        
        finally:
            signal.alarm(0)
    torch.save(valid_data_list, f"./data_store/valid_{task_id}.pt")
    print("save valid_data_list")
    
    test_smiles_list = split['test'][smiles_name].tolist()
    test_label_list = split['test']['Y'].tolist()
    test_data_list = []
    for idx in range(len(test_smiles_list)):
        try:
            signal.alarm(600)
            mol_tmp = Chem.MolFromSmiles(test_smiles_list[idx])
            if mol_tmp.GetNumHeavyAtoms() > 200:
                print(f"The mol is too large(>200): {test_smiles_list[idx]}")
                continue
            mol_smiles = Chem.MolToSmiles(mol_tmp)
            data_item = smiles_preporcess(mol_smiles, f"./data_store/{task_id}_mol_sdf/mol_{mol_smiles.replace('/', '_')[:245]}.sdf")
            data_item['label'] = torch.tensor(test_label_list[idx])
            data_item['task_id'] = task_id
            test_data_list.append(data_item)
                        
        except TimeoutException:
            print(f"Timeout : {test_smiles_list[idx]}")
            continue
                        
        except Exception as e:
            print(f"Error: {test_smiles_list[idx]} | {e}")
            continue
                        
        finally:
            signal.alarm(0)
    torch.save(test_data_list, f"./data_store/test_{task_id}.pt")
    print("save test_data_list: ", len(train_data_list), len(valid_data_list), len(test_data_list))
            
import argparse
import torch.multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=str)
    args = parser.parse_args()
    get_dataset(args.task_id)
