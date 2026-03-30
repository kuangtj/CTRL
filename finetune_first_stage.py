import os
import shutil
import sys
import yaml
import numpy as np
from datetime import datetime

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, average_precision_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from dataset.CTRL_finetune_first_stage_dataset import MolTestDatasetWrapper
import math

def calc_rmse(labels, preds, valid):
    """tbd"""
    return torch.sqrt(torch.mean((preds - labels) ** 2))

def calc_mae(labels, preds, valid):
    """tbd"""
    return torch.mean(torch.abs(preds - labels))

def calc_rmse_np(labels, preds, valid):
    """tbd"""
    return np.sqrt(np.mean((preds - labels) ** 2))

def calc_mae_np(labels, preds, valid):
    """tbd"""
    return np.mean(np.abs(preds - labels))

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

task_weight_list = {
    "oncopolypharmacology": 500,
    "gdsc2": 18.5,
    "gdsc1": 21,
    "buchwald-hartwig": 1,
    "kiba": 17,
    "davis": 5,
    "bindingdb_kd": 8,
    "lipophilicity": 6,
    "esol": 13,
    "clearance_microsome_az": 147,
    "clearance_hepatocyte_az": 147,
    "half_life_obach": 1200,
    "vdss_lombardo": 700,
    "ld50_zhu": 10.5,
    "ppbr_az": 90,
    "solubility_aqsoldb": 15.5,
    "lipophilicity_astrazeneca": 6,
    "caco2_wang": 4,
}


def calc_rocauc_score(labels, preds, valid):
    """compute ROC-AUC and averaged across tasks"""
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
        valid = valid.reshape(-1, 1)

    rocauc_list = []

    if valid.ndim == 3:
        valid = valid.squeeze(-1)
    for i in range(labels.shape[1]):
        c_valid = valid[:, i].astype("bool")
        c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
        #AUC is only defined when there is at least one positive data.
        if len(np.unique(c_label)) == 2:
            rocauc_list.append(roc_auc_score(c_label, c_pred))

    print('Valid ratio: %s' % (np.mean(valid)))
    print('Task evaluated: %s/%s' % (len(rocauc_list), labels.shape[1]))
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list)/len(rocauc_list)
    
def calc_rmse(labels, preds):
    """tbd"""
    return torch.sqrt(torch.mean((preds - labels) ** 2))

import torchmetrics
class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        self.dataset = dataset
        self.BCELoss = nn.BCELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.L1Loss = nn.L1Loss()
        self.MSELoss = nn.MSELoss()
        self.PearsonCorrCoef = torchmetrics.PearsonCorrCoef().to(self.device)
        print("FineTune init done")

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device
    
    def _step(self, model, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels, val=False, epoch_counter=0):
        pred = model(data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, epoch_counter)
        weight_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'hiv': [], 'clintox': [], 'sider': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}
        tmp_loss_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'hiv': [], 'clintox': [], 'sider': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}

        total_loss = 0
        label_start = 0

        total_loss = 0

        for idx in range(len(pred)):
            
            task_id = tasks[idx]
            pred_item = pred[idx]
            if task_id == 'clintox':
                label_item = labels[label_start:label_start + 2]
                label_start += 2
            elif task_id == 'sider':
                label_item = labels[label_start:label_start + 27]
                label_start += 27
            elif task_id == 'drugbank':
                label_item = labels[label_start:label_start + 86]
                label_start += 86
            else:
                label_item = labels[label_start:label_start + 1]
                label_start += 1

           
            if task_id in ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp3a4_substrate_carbonmangels', 'herg', 'ames', 'dili', 'cyp2d6_veith', 'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels', 'drugbank', 'bbbp', 'hiv', 'clintox', 'sider']:
                loss_tmp = self.BCELoss(pred_item, label_item.float())

            if task_id in ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'ppbr_az', 'ld50_zhu', 'vdss_lombardo', 'half_life_obach', 'clearance_hepatocyte_az', 'clearance_microsome_az', 'davis', 'kiba', 'bindingdb_kd', 'gdsc1', 'gdsc2', 'oncopolypharmacology', 'buchwald-hartwig', 'esol', 'lipophilicity']:
                loss_tmp = calc_rmse(label_item.float(), pred_item) / math.sqrt(task_weight_list[task_id])
                
            tmp_loss_list[task_id].append(loss_tmp)
            
            total_loss += loss_tmp

        assert label_start == len(labels)
                
        return total_loss/len(morgan_fp), pred, tmp_loss_list

    def _step2(self, model, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels, val=False, epoch_counter=0):
        pred = model(data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, epoch_counter)
        weight_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'hiv': [], 'clintox': [], 'sider': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}
        tmp_loss_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'hiv': [], 'clintox': [], 'sider': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}

        total_loss = 0
        label_start = 0


        for idx in range(len(pred)):
            
            task_id = tasks[idx]
            pred_item = pred[idx]
            label_item = labels[idx]
            if task_id == 'clintox':
                label_item = labels[label_start:label_start + 2]
                label_start += 2
            elif task_id == 'sider':
                label_item = labels[label_start:label_start + 27]
                label_start += 27
            elif task_id == 'drugbank':
                label_item = labels[label_start:label_start + 86]
                label_start += 86
            else:
                label_item = labels[label_start:label_start + 1]
                label_start += 1
                
            if task_id in ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp3a4_substrate_carbonmangels', 'herg', 'ames', 'dili', 'cyp2d6_veith', 'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels', 'drugbank', 'bbbp', 'hiv', 'clintox', 'sider']:
                bad = (~torch.isfinite(label_item)) | (label_item < 0) | (label_item > 1)
                loss_tmp = self.BCELoss(pred_item, label_item.float())

            if task_id in ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'ppbr_az', 'ld50_zhu', 'vdss_lombardo', 'half_life_obach', 'clearance_hepatocyte_az', 'clearance_microsome_az', 'davis', 'kiba', 'bindingdb_kd', 'gdsc1', 'gdsc2', 'oncopolypharmacology', 'buchwald-hartwig', 'esol', 'lipophilicity']:
                loss_tmp = calc_rmse(label_item.float(), pred_item) / math.sqrt(task_weight_list[task_id])
                
            tmp_loss_list[task_id].append(loss_tmp)
            total_loss += loss_tmp

        assert label_start == len(labels)
                
        return total_loss/len(morgan_fp), pred, tmp_loss_list

    def train(self):
            
        if self.config['model_type'] == "norm":
            from models.finetune_model import Mol_Encoder_norm
            encoder = Mol_Encoder_norm(hidden_dim=self.config['dim'], num_transformer_layers=self.config['layer'], num_heads=8, device=self.device)
            from models.finetune_model import finetune_model_norm_all_task_together
            model = finetune_model_norm_all_task_together(hidden_dim=self.config['dim'], tf_layer_num=self.config['layer'], num_heads=8, tasks_info=config['task_info'], encoder=encoder).to(self.device)
            
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders(self.config['model_type'])
            
        base_params = [p for n,p in model.named_parameters() if 'pred_head' not in n]
        pred_params = [p for n,p in model.named_parameters() if 'pred_head' in n]
        base_params_num = 0
        pred_params_num = 0
        for i in base_params:
            base_params_num += i.numel()
        for i in pred_params:
            pred_params_num += i.numel()
        
        print("base_params: ", base_params_num)
        print("pred_params: ", pred_params_num)
        
        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['lr']}, {'params': pred_params, 'lr': self.config['lr']}],
            self.config['init_lr'], weight_decay=self.config['weight_decay']
        )

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0
        best_test = 100
        best_result = 0
        best_loss = 1000
        train_loss_list = []
        valid_loss_list = []
        test_loss_list = []

        for epoch_counter in range(self.config['warm_up']):
            import time
            start_time = time.time()
            train_loss = 0.0
            num_data = 0
            all_tmp_loss_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'hiv': [], 'clintox': [], 'sider': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}
            print('epoch_counter: ', epoch_counter)
            for bn, [data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels] in enumerate(train_loader):
                if (bn + 0) % 10 == 0:
                    print("----------", bn, "----------")
                    print("lr: ", f"{optimizer.param_groups[0]['lr']:.8f}", f"{optimizer.param_groups[1]['lr']:.8f}")
                optimizer.zero_grad()

                data_2D = data_2D.to(self.device)
                data_3D1 = data_3D1.to(self.device)
                data_3D2 = data_3D2.to(self.device)
                morgan_fp = morgan_fp.to(self.device)
                maccs_fp = maccs_fp.to(self.device)
                daylight_fp = daylight_fp.to(self.device)
                descriptor_int = descriptor_int.to(self.device)
                descriptor_float = descriptor_float.to(self.device)
                src_mask = src_mask.to(self.device)
                mask_3d = mask_3d.to(self.device)
                if len(prot_rep) > 0:
                    prot_rep = prot_rep.to(self.device)
                if len(cellline_drp_rep) > 0:
                    cellline_drp_rep = cellline_drp_rep.to(self.device)
                if len(cellline_dsp_rep) > 0:
                    cellline_dsp_rep = cellline_dsp_rep.to(self.device)
                labels = labels.to(self.device)

                loss, _, tmp_loss_list = self._step(model, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels, epoch_counter=epoch_counter)
                for loss_key, loss_value in tmp_loss_list.items():
                    if len(loss_value) == 0:
                        continue
                    all_tmp_loss_list[loss_key] += loss_value

                loss.backward()

                train_loss += loss.item() * len(labels)
                num_data += len(labels)
                optimizer.step()
                
                n_iter += 1
                del data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels, loss, _

                # validate the model if requested
                if (bn + 1) % int(len(train_loader)/20) == 0:
                    for loss_key, loss_value in all_tmp_loss_list.items():
                        if len(loss_value) == 0:
                            continue
                        print(loss_key, sum(loss_value)/len(loss_value))
                    valid_loss, valid_results = self._validate(model, valid_loader, epoch_counter)
                    print('Valid loss:', valid_loss)
                    print('Valid result: ', valid_results)
                    valid_loss_list.append(valid_loss)
                    test_loss, test_results = self._validate(model, test_loader, epoch_counter, test=True)
                    print('Test loss:', test_loss)
                    print('Test result: ', test_results)
                    test_loss_list.append(test_loss)
                    if best_loss > valid_loss:
                        best_loss = valid_loss
                        best_result = test_results

                    print("best_loss: ", best_loss)
                    print("best_result: ", best_result)

                    torch.save(model.state_dict(), f"./ckpt_finetune/{self.config['model_type']}/epoch_{epoch_counter}_{valid_n_iter}_weight.pth")
                    print("valid_n_iter: ", valid_n_iter)
                    valid_n_iter += 1

            train_loss /= num_data
            train_loss_list.append(train_loss)
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] - 0.05 * self.config['lr']
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] - 0.05 * self.config['lr']
         
    def _validate(self, model, valid_loader, epoch_counter, test=False):
        predictions_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'clintox': [], 'sider': [], 'hiv': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}
        weight_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'clintox': [], 'sider': [], 'hiv': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}
        labels_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'clintox': [], 'sider': [], 'hiv': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}
        valids_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'clintox': [], 'sider': [], 'hiv': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}
        result_list = {'hia_hou': None, 'pgp_broccatelli': None, 'bioavailability_ma': None, 'bbb_martins': None, 'cyp3a4_substrate_carbonmangels': None, 'herg': None, 'ames': None, 'dili': None, 'cyp2d6_veith': None, 'cyp3a4_veith': None, 'cyp2c9_veith': None, 'cyp2d6_substrate_carbonmangels': None, 'cyp2c9_substrate_carbonmangels': None, 'caco2_wang': None, 'lipophilicity_astrazeneca': None, 'solubility_aqsoldb': None, 'ppbr_az': None, 'ld50_zhu': None, 'vdss_lombardo': None, 'half_life_obach': None, 'clearance_hepatocyte_az': None, 'clearance_microsome_az': None, 'bbbp': None, 'clintox': None, 'sider': None, 'hiv': None, 'esol': None, 'lipophilicity': None, 'davis': None, 'kiba': None, 'bindingdb_kd': None, 'drugbank': None, 'buchwald-hartwig': None, 'gdsc1': None, 'gdsc2': None, 'oncopolypharmacology': None}
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            all_tmp_loss_list = {'hia_hou': [], 'pgp_broccatelli': [], 'bioavailability_ma': [], 'bbb_martins': [], 'cyp3a4_substrate_carbonmangels': [], 'herg': [], 'ames': [], 'dili': [], 'cyp2d6_veith': [], 'cyp3a4_veith': [], 'cyp2c9_veith': [], 'cyp2d6_substrate_carbonmangels': [], 'cyp2c9_substrate_carbonmangels': [], 'caco2_wang': [], 'lipophilicity_astrazeneca': [], 'solubility_aqsoldb': [], 'ppbr_az': [], 'ld50_zhu': [], 'vdss_lombardo': [], 'half_life_obach': [], 'clearance_hepatocyte_az': [], 'clearance_microsome_az': [], 'bbbp': [], 'hiv': [], 'clintox': [], 'sider': [], 'esol': [], 'lipophilicity': [], 'davis': [], 'kiba': [], 'bindingdb_kd': [], 'drugbank': [], 'buchwald-hartwig': [], 'gdsc1': [], 'gdsc2': [], 'oncopolypharmacology': []}
            for bn, [data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels] in enumerate(valid_loader):
                if bn % 10 == 0:
                    print("----------", bn, "----------")
                data_2D = data_2D.to(self.device)
                data_3D1 = data_3D1.to(self.device)
                data_3D2 = data_3D2.to(self.device)
                morgan_fp = morgan_fp.to(self.device)
                maccs_fp = maccs_fp.to(self.device)
                daylight_fp = daylight_fp.to(self.device)
                descriptor_int = descriptor_int.to(self.device)
                descriptor_float = descriptor_float.to(self.device)
                src_mask = src_mask.to(self.device)
                mask_3d = mask_3d.to(self.device)
                if len(prot_rep) > 0:
                    prot_rep = prot_rep.to(self.device)
                if len(cellline_drp_rep) > 0:
                    cellline_drp_rep = cellline_drp_rep.to(self.device)
                if len(cellline_dsp_rep) > 0:
                    cellline_dsp_rep = cellline_dsp_rep.to(self.device)
                labels = labels.to(self.device)

                loss, pred, tmp_loss_list = self._step2(model, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels, val=True, epoch_counter=epoch_counter)

                for loss_key, loss_value in tmp_loss_list.items():
                    if len(loss_value) == 0:
                        continue
                    all_tmp_loss_list[loss_key] += loss_value

                label_start = 0
                for idx in range(len(tasks)):
                    predictions_list[tasks[idx]].append(pred[idx].detach().cpu().numpy())
                    
                    if tasks[idx] == 'clintox':
                        label_item_tmp = labels[label_start:label_start + 2]
                        label_start += 2
                    elif tasks[idx] == 'sider':
                        label_item_tmp = labels[label_start:label_start + 27]
                        label_start += 27
                    elif tasks[idx] == 'drugbank':
                        label_item_tmp = labels[label_start:label_start + 86]
                        label_start += 86
                    else:
                        label_item_tmp = labels[label_start:label_start + 1]
                        label_start += 1
                    labels_list[tasks[idx]].append(label_item_tmp.cpu().numpy())

                valid_loss += loss.item() * len(labels)
                num_data += len(labels)
                del data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels, loss, pred

            valid_loss /= num_data

            for loss_key, loss_value in all_tmp_loss_list.items():
                if len(loss_value) == 0:
                    continue
                print(loss_key, sum(loss_value)/len(loss_value))

        for task_id, predictions in predictions_list.items():

            if task_id in ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp3a4_substrate_carbonmangels', 'herg', 'ames', 'dili', 'drugbank']:
                predictions = np.array(predictions).reshape(-1, 1)
                labels = np.array(labels_list[task_id]).reshape(-1, 1)
                AUROC = roc_auc_score(labels, predictions)
                result_list[task_id] = AUROC

            if task_id == 'bbbp' or task_id == 'hiv' or task_id == 'clintox' or task_id == 'sider':
                predictions = np.array(predictions)
                pred_classes = (predictions >= 0.5).astype(int).reshape(-1)
                labels = np.array(labels_list[task_id]).astype(int).reshape(-1)
                roc_auc = accuracy_score(labels, pred_classes)
                result_list[task_id] = roc_auc

            if task_id in ['cyp2d6_veith', 'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels']:
                predictions = np.array(predictions).reshape(-1)
                labels = np.array(labels_list[task_id]).reshape(-1)
                AUPRC = average_precision_score(labels, predictions)
                result_list[task_id] = AUPRC

            if task_id in ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'ppbr_az', 'ld50_zhu']:
                predictions = np.array(predictions)
                labels = np.array(labels_list[task_id])
                MAE = mean_absolute_error(labels, predictions)
                result_list[task_id] = MAE

            if task_id in ['vdss_lombardo', 'half_life_obach', 'clearance_hepatocyte_az', 'clearance_microsome_az']:
                predictions = np.array(predictions)
                labels = np.array(labels_list[task_id])
                if len(labels.shape) > 1:
                    labels = labels[:, 0]
                Spearman, _ = spearmanr(labels, predictions[:, 0])
                result_list[task_id] = Spearman

            if task_id in ['bindingdb_kd', 'buchwald-hartwig', 'gdsc1', 'gdsc2', 'oncopolypharmacology']:
                predictions = np.array(predictions)
                labels = np.array(labels_list[task_id])
                if len(labels.shape) > 1:
                    labels = labels[:, 0]
                PCC, _ = pearsonr(labels, predictions[:, 0])
                result_list[task_id] = PCC

            if task_id in ['davis', 'kiba']:
                predictions = np.array(predictions)
                labels = np.array(labels_list[task_id])
                MSE = mean_squared_error(labels, predictions)
                result_list[task_id] = MSE

            if task_id in ['esol', 'lipophilicity']:
                predictions = torch.tensor(np.array(predictions)).to(self.device)
                labels = torch.tensor(np.array(labels_list[task_id])).to(self.device)
                RMSE = calc_rmse(labels, predictions).item()
                result_list[task_id] = RMSE
        
        model.train()

        return valid_loss, result_list

def main(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], config['num_workers'])
    fine_tune = FineTune(dataset, config)
    fine_tune.train()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--init_lr', type=float, default=0.0004)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    args = parser.parse_args()

    config = {}

    config['gpu'] = args.device
    config['model_type'] = args.model
    config['layer'] = args.layer
    config['dim'] = args.dim
    config['warm_up'] = args.warm_up
    config['lr'] = args.lr
    config['init_lr'] = args.init_lr
    config['weight_decay'] = args.weight_decay
    config['num_workers'] = args.num_workers

    task_info = {
        'hia_hou': ["hia_hou"],
        'pgp_broccatelli': ["pgp_broccatelli"],
        'bioavailability_ma': ["bioavailability_ma"],
        'bbb_martins': ["bbb_martins"],
        'cyp3a4_substrate_carbonmangels': ["cyp3a4_substrate_carbonmangels"],
        'herg': ["herg"],
        'ames': ["ames"],
        'dili': ["dili"],
        'cyp2d6_veith': ["cyp2d6_veith"],
        'cyp3a4_veith': ["cyp3a4_veith"],
        'cyp2c9_veith': ["cyp2c9_veith"],
        'cyp2d6_substrate_carbonmangels': ["cyp2d6_substrate_carbonmangels"],
        'cyp2c9_substrate_carbonmangels': ["cyp2c9_substrate_carbonmangels"],
        'caco2_wang': ["caco2_wang"],
        'lipophilicity_astrazeneca': ["lipophilicity_astrazeneca"],
        'solubility_aqsoldb': ["solubility_aqsoldb"],
        'ppbr_az': ["ppbr_az"],
        'ld50_zhu': ["ld50_zhu"],
        'vdss_lombardo': ["vdss_lombardo"],
        'half_life_obach': ["half_life_obach"],
        'clearance_hepatocyte_az': ["clearance_hepatocyte_az"],
        'clearance_microsome_az': ["clearance_microsome_az"],
        'bbbp': ["p_np"],
        'clintox': ['CT_TOX', 'FDA_APPROVED'],
        'hiv': ["HIV_active"],
        'sider': ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", "Reproductive system and breast disorders", "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", "General disorders and administration site conditions", "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders", "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders", "Infections and infestations", "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders", "Nervous system disorders", "Injury, poisoning and procedural complications"],
        'esol': ["measured log solubility in mols per litre"],
        'lipophilicity': ["exp"],
        'davis': ["davis"],
        'kiba': ["kiba"],
        'bindingdb_kd': ["bindingdb_kd"],
        'drugbank': ["drugbank"],
        'buchwald-hartwig': ["buchwald-hartwig"],
        'gdsc1': ["gdsc1"],
        'gdsc2': ["gdsc2"],
        'oncopolypharmacology': ["oncopolypharmacology"],
    }

    config['task_info'] = task_info

    print("config: ", config)

    result = main(config)
