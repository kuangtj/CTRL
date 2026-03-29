
# CTRL Code Repository

This repository contains the official codebase and implementation for the CTRL research paper.

## Requirements

The codebase is built and tested on **Python 3.9**. We strongly recommend using [Conda](https://docs.conda.io/en/latest/) to manage your virtual environment and install dependencies to avoid conflicts, especially with CUDA-related packages.

### Major Dependencies

- **Deep Learning & Graph Neural Networks:** `torch` (v1.12.1+cu113 recommended), `torch-geometric`, `dgl`
- **Cheminformatics & Bioinformatics:** `rdkit`, `PyTDC`, `openbabel`, `biopython`
- **Large Language Models (LLMs):** `transformers`, `accelerate`
- **Data Science & Utility:** `numpy`, `pandas`, `scipy`, `scikit-learn`

---

## Installation

**Step 1: Create and activate a new Conda environment**

```bash
conda create -n ctrl_env python=3.9
conda activate ctrl_env
```

### Install PyTorch

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Install PyG dependencies

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric
```

### Install Core Cheminformatics Tools

```bash
conda install -c conda-forge rdkit openbabel
```

### Install Remaining Python Packages

```bash
pip install PyTDC transformers accelerate dgl biopython numpy pandas scikit-learn tqdm
```

---

## Data Preprocessing

### For Pretraining:

```bash
python ./dataset/get_pretrain_store.py --dataset {dataset_id}
```

`dataset_id` can be either `pubchem` or `zinc`.

### For Chemical Tasks:

#### For MoleculeNet Data:

```bash
python ./dataset/get_MolNet_store.py --task_id {task_id}
```

`task_id` can be one of the following: `bbbp`, `clintox`, `hiv`, `sider`, `esol`, `lipophilicity`.

#### For TDC Dataset:

##### For Single-instance Tasks:

```bash
python ./dataset/get_TDC_store.py --task_id {task_id}
```

`task_id` can be one of the following: `ames`, `bbb_martins`, `bioavailability_ma`, `caco2_wang`, `clearance_hepatocyte_az`, `clearance_microsome_az`, `dili`, `herg`, `hia_hou`, `id50`, `lipophilicity_astrazeneca`, `pgp_broccatelli`, `ppbr_az`, `solubility_aqsoldb`, `vdss_lombardo`, `cyp2d6_veith`, `cyp3a4_veith`, `cyp2c9_veith`, `cyp3a4_substrate_carbonmangels`, `cyp2d6_substrate_carbonmangels`, `cyp2c9_substrate_carbonmangels`.

##### For Drug-Target Interaction Prediction:

```bash
python get_DTI_mol.py
python get_DTI_prot.py
python get_DTI_store.py --task_id {task_id}
python get_DTI_prot_store.py --task_id {task_id} --device {your_device}
```

`task_id` can be one of the following: `davis`, `kiba`, `bindingdb_kd`.

##### For Drug-Drug Interaction Prediction:

```bash
python get_DDI_mol.py
python get_DDI_prot.py --task_id {task_id}
```

`task_id` can be `drugbank`.

##### For Drug Response Prediction:

```bash
python get_DRP_mol.py
python get_DRP_prot.py --task_id {task_id}
```

`task_id` can be one of the following: `gdsc1`, `gdsc2`.

##### For Drug Synergy Prediction:

```bash
python get_DSP_mol.py
python get_DSP_prot.py --task_id {task_id}
```

`task_id` can be `oncopolypharmacology`.

##### For Reaction Yields Prediction:

```bash
python get_RYP_mol.py
python get_RYP_prot.py --task_id {task_id}
```

`task_id` can be `buchwald-hartwig`.

---

## Pretraining

```bash
python pretrain_stage.py --device {your_device}
```

## Fine-tuning

### For All Tasks (First Stage Training):

```bash
python finetune_first_stage.py --device {your_device}
```

### For Fine-tuning the `pred_head` (Second Stage Training):

```bash
python finetune_second_stage.py --device {your_device}
```

---

More information will be provided soon!!!
