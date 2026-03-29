from tdc.multi_pred import DTI
import pandas as pd
from rdkit import Chem

data_davis = DTI(name='DAVIS', path='./data/DTI_data_from_TDC')
df_davis = data_davis.get_data()

data_kiba = DTI(name='KIBA', path='./data/DTI_data_from_TDC')
df_kiba = data_kiba.get_data()

data_bdb = DTI(name='BindingDB_Kd', path='./data/DTI_data_from_TDC')
df_bdb = data_bdb.get_data()

smiles_davis = df_davis['Drug'].tolist()
smiles_kiba = df_kiba['Drug'].tolist()
smiles_bdb = df_bdb['Drug'].tolist()
ID_davis = df_davis['Drug_ID'].tolist()
ID_kiba = df_kiba['Drug_ID'].tolist()
ID_bdb = df_bdb['Drug_ID'].tolist()

davis_name_smiles_all = []
for i in range(len(smiles_davis)):
    smiles_item = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_davis[i]))
    davis_name_smiles_all.append((str(ID_davis[i]) + str(smiles_item.replace('/', '_')[:245 - len(str(ID_davis[i]))]), smiles_item))
davis_name_smiles_all = list(set(davis_name_smiles_all))

kiba_name_smiles_all = []
for i in range(len(smiles_kiba)):
    smiles_item = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_kiba[i]))
    kiba_name_smiles_all.append((str(ID_kiba[i]) + str(smiles_item.replace('/', '_')[:245 - len(str(ID_kiba[i]))]), smiles_item))
kiba_name_smiles_all = list(set(kiba_name_smiles_all))
    
bdb_name_smiles_all = []
for i in range(len(smiles_bdb)):
    smiles_item = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_bdb[i]))
    bdb_name_smiles_all.append((str(ID_bdb[i]) + str(smiles_item.replace('/', '_')[:245 - len(str(ID_bdb[i]))]), smiles_item))
bdb_name_smiles_all = list(set(bdb_name_smiles_all))
    
import json

data_path = "./data/DTI_data_from_TDC/davis_data.json"
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(davis_name_smiles_all, f, indent=4)

data_path = "./data/DTI_data_from_TDC/kiba_data.json"
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(kiba_name_smiles_all, f, indent=4)

data_path = "./data/DTI_data_from_TDC/bdb_data.json"
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(bdb_name_smiles_all, f, indent=4)




from tdc.multi_pred import DTI
import pandas as pd
from rdkit import Chem

data_davis = DTI(name='DAVIS', path='./data/DTI_data_from_TDC')
df_davis = data_davis.get_data()

data_kiba = DTI(name='KIBA', path='./data/DTI_data_from_TDC')
df_kiba = data_kiba.get_data()

data_bdb = DTI(name='BindingDB_Kd', path='./data/DTI_data_from_TDC')
df_bdb = data_bdb.get_data()

smiles_davis = df_davis['Drug'].tolist()
smiles_kiba = df_kiba['Drug'].tolist()
smiles_bdb = df_bdb['Drug'].tolist()
ID_davis = df_davis['Drug_ID'].tolist()
ID_kiba = df_kiba['Drug_ID'].tolist()
ID_bdb = df_bdb['Drug_ID'].tolist()

smiles_all = smiles_davis + smiles_bdb + smiles_kiba

import json

data_path = "./data/DTI_data_from_TDC/DTI_data.json"
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(smiles_all, f, indent=4)




