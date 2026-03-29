from tdc.multi_pred import DTI
import pandas as pd
from rdkit import Chem

data_davis = DTI(name='DAVIS', path='./data/DTI_data_from_TDC')
df_davis = data_davis.get_data()

data_kiba = DTI(name='KIBA', path='./data/DTI_data_from_TDC')
df_kiba = data_kiba.get_data()

data_bdb = DTI(name='BindingDB_Kd', path='./data/DTI_data_from_TDC')
df_bdb = data_bdb.get_data()

Target_davis = df_davis['Target'].tolist()
Target_kiba = df_kiba['Target'].tolist()
Target_bdb = df_bdb['Target'].tolist()
Target_ID_davis = df_davis['Target_ID'].tolist()
Target_ID_kiba = df_kiba['Target_ID'].tolist()
Target_ID_bdb = df_bdb['Target_ID'].tolist()

Target_davis = list(set(Target_davis))
Target_kiba = list(set(Target_kiba))
Target_bdb = list(set(Target_bdb))

davis_name_target_all = []
prot_idx = 0
for i in range(len(Target_davis)):
    name_item = str(prot_idx) + "_" + str(Target_davis[i][:240])
    davis_name_target_all.append((name_item, Target_davis[i]))
    prot_idx = prot_idx + 1
davis_name_target_all = list(set(davis_name_target_all))

kiba_name_target_all = []
for i in range(len(Target_kiba)):
    name_item = str(prot_idx) + "_" + str(Target_kiba[i][:240])
    kiba_name_target_all.append((name_item, Target_kiba[i]))
    prot_idx = prot_idx + 1
kiba_name_target_all = list(set(kiba_name_target_all))
    
bdb_name_target_all = []
for i in range(len(Target_bdb)):
    name_item = str(prot_idx) + "_" + str(Target_bdb[i][:240])
    bdb_name_target_all.append((name_item, Target_bdb[i]))
    prot_idx = prot_idx + 1
bdb_name_target_all = list(set(bdb_name_target_all))

import json

data_path = "./data/DTI_data_from_TDC/davis_prot_data.json"
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(davis_name_target_all, f, indent=4)

data_path = "./data/DTI_data_from_TDC/kiba_prot_data.json"
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(kiba_name_target_all, f, indent=4)

data_path = "./data/DTI_data_from_TDC/bdb_prot_data.json"
with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(bdb_name_target_all, f, indent=4)




