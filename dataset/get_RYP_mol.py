from tdc.single_pred import Yields
import pandas as pd

data = Yields(name='Buchwald-Hartwig', path='./data/RYP_data_from_TDC')
df = data.get_data()

smiles_list = []
for i in df['Reaction']:
    smiles_list.append(i['product'])
    smiles_list.append(i['reactant'])
    if i['catalyst'] != '':
        smiles_list.append(i['catalyst'])
from rdkit import Chem
unique_smiles_list = list(set(smiles_list))
import json

output_file = "./data/RYP_data_from_TDC/buchwald-hartwig_data.json"

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(unique_smiles_list, f, indent=4)
