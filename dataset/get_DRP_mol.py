from tdc.multi_pred import DrugRes
import pandas as pd
from rdkit import Chem

data_gdsc1 = DrugRes(name='GDSC1', path='./data/DRP_data_from_TDC')
df_gdsc1 = data_gdsc1.get_data()

data_gdsc2 = DrugRes(name='GDSC2', path='./data/DRP_data_from_TDC')
df_gdsc2 = data_gdsc2.get_data()

smiles_gdsc1_tmp = df_gdsc1['Drug'].tolist()
smiles_gdsc2_tmp = df_gdsc2['Drug'].tolist()
smiles_gdsc1 = list(set(smiles_gdsc1_tmp))
smiles_gdsc2 = list(set(smiles_gdsc2_tmp))

import json
gdsc1_path = "./data/DRP_data_from_TDC/gdsc1_data.json"
with open(gdsc1_path, 'w', encoding='utf-8') as f:
    json.dump(smiles_gdsc1, f, indent=4)

gdsc2_path = "./data/DRP_data_from_TDC/gdsc2_data.json"
with open(gdsc2_path, 'w', encoding='utf-8') as f:
    json.dump(smiles_gdsc2, f, indent=4)