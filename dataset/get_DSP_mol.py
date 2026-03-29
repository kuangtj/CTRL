from tdc.multi_pred import DrugSyn
import pandas as pd
from rdkit import Chem

data = DrugSyn(name='OncoPolyPharmacology', path='./data/DSP_data_from_TDC')
df = data.get_data()

smiles_list_1 = df['Drug1'].tolist()
smiles_list_2 = df['Drug2'].tolist()

unique_smiles_list_tmp = list(set(smiles_list_1 + smiles_list_2))
unique_smiles_list = []
for i in unique_smiles_list_tmp:
    unique_smiles_list.append(i)
unique_smiles_list = list(set(unique_smiles_list))

import json
data_path = "./data/DSP_data_from_TDC/oncopolypharmacology_data.json"

with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(unique_smiles_list, f, indent=4)