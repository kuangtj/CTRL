from tdc.multi_pred import DDI

data = DDI(name='DrugBank', path='./data/DDI_data_from_TDC')

df = data.get_data()

smiles_list_1 = df['Drug1'].tolist()
smiles_list_2 = df['Drug2'].tolist()
from rdkit import Chem
unique_smiles_list = list(set(smiles_list_1 + smiles_list_2))
output_file = "./data/DDI_data_from_TDC/drugbank_data.json"
import json
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(unique_smiles_list, f, indent=4)
