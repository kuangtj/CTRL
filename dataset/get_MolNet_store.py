import pandas as pd
import os

datasets = {
    './data/ADMET_data_from_MolNet/bbbp/BBBP.csv': ["p_np"],
    './data/ADMET_data_from_MolNet/clintox/clintox.csv': ['CT_TOX', 'FDA_APPROVED'],
    './data/ADMET_data_from_MolNet/hiv/HIV.csv': ["HIV_active"],
    './data/ADMET_data_from_MolNet/sider/sider.csv': [
        "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
        "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
        "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
        "Reproductive system and breast disorders", "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
        "General disorders and administration site conditions", "Endocrine disorders", "Surgical and medical procedures", 
        "Vascular disorders", "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
        "Congenital, familial and genetic disorders", "Infections and infestations", "Respiratory, thoracic and mediastinal disorders", 
        "Psychiatric disorders", "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
        "Ear and labyrinth disorders", "Cardiac disorders", "Nervous system disorders", "Injury, poisoning and procedural complications"
    ],
    './data/ADMET_data_from_MolNet/esol/esol.csv': ["measured log solubility in mols per litre"],
    './data/ADMET_data_from_MolNet/lipophilicity/Lipophilicity.csv': ["exp"]
}

from data_process import smiles_preporcess
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

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

import os
import torch
import pandas as pd
from rdkit import Chem
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout !!!")

def get_dataset(task_id_want):
    for file_name, labels in datasets.items():
        task_id = os.path.basename(file_name)[:-4].lower()
        if task_id != task_id_want:
            continue
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
        
            if 'smiles' in df.columns:
                missing_cols = [col for col in labels if col not in df.columns]
                if missing_cols:
                    continue
            
                smiles_list_all = df['smiles'].tolist()
            
                if len(labels) == 1:
                    label_list_all = df[labels[0]].tolist()
                else:
                    label_list_all = df[labels].values.tolist()

                label_list = label_list_all
                smiles_list = smiles_list_all
            
                all_data_list = []
                
                signal.signal(signal.SIGALRM, timeout_handler)
                
                for idx in range(len(smiles_list)):
                    try:
                        signal.alarm(600) 
                        mol_tmp = Chem.MolFromSmiles(smiles_list[idx])
                        if mol_tmp.GetNumHeavyAtoms() > 200:
                            print(f"The mol is too large(>200): {smiles_list[idx]}")
                        mol_smiles = Chem.MolToSmiles(mol_tmp)
                        
                        data_item = smiles_preporcess(mol_smiles, f"./data_store/{task_id}_mol_sdf/mol_{mol_smiles.replace('/', '_')[:245]}.sdf")
                        data_item['label'] = torch.tensor(label_list[idx])
                        data_item['task_id'] = task_id
                        all_data_list.append(data_item)
                        
                    except TimeoutException:
                        print(f"Timeout: {smiles_list[idx]}")
                        continue
                        
                    except Exception as e:
                        print(f"Error: {smiles_list[idx]} | {e}")
                        continue
                        
                    finally:
                        signal.alarm(0)

                train_data_items_pro, valid_data_items_pro, test_data_items_pro = scaffold_split(all_data_list, 0.1, 0.2)
                torch.save(train_data_items_pro, f"./data_store/train_{task_id}.pt")
                torch.save(valid_data_items_pro, f"./data_store/valid_{task_id}.pt")
                torch.save(test_data_items_pro, f"./data_store/test_{task_id}.pt")
                print("save test_data_list: ", len(train_data_items_pro), len(valid_data_items_pro), len(test_data_items_pro))
           
            else:
                print(f"smiles col is not found in {file_name}")
        else:
            print(f"{file_name} is not found")
            
import argparse
import torch.multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=str)
    args = parser.parse_args()
    get_dataset(args.task_id)
