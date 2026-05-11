import os
import shutil
import sys
import yaml
import numpy as np
#import pandas as pd
from datetime import datetime

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, average_precision_score, accuracy_score
from torchmetrics import MeanSquaredError
from scipy.stats import pearsonr, spearmanr
#from dataset.CLP_dataset import MolTestDatasetWrapper, MolTestDatasetWrapper_7
from dataset.CTRL_inference_dataset import MolTestDatasetWrapper
import math
apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

def calc_rmse(labels, preds, valid):
    """tbd"""
#    print("preds: ", preds.shape)
#    print("labels: ", labels.shape)
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

def calc_rocauc_score(labels, preds, valid):
    """compute ROC-AUC and averaged across tasks"""
#    print("labels: ", labels.shape)
#    print("preds: ", preds.shape)
#    print("valid: ", valid.shape)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
        valid = valid.reshape(-1, 1)
#        labels = np.expand_dims(labels, axis=-1)
#        preds = np.expand_dims(preds, axis=-1)
#        valid = np.expand_dims(valid, axis=-1)
#    print(labels.shape)
#    print(preds.shape)
#    print(valid.shape)

    rocauc_list = []
#    print("labels: ", labels.shape)
#    print("preds: ", preds.shape)
#    print("valid: ", valid.shape)
#    print("labels: ", labels.shape)
#    print("preds: ", preds.shape)
#    print("valid: ", valid.shape)
    if valid.ndim == 3:
        valid = valid.squeeze(-1)
    for i in range(labels.shape[1]):
#        print("valid: ", valid)
#        print("i: ", i)
        c_valid = valid[:, i].astype("bool")
#        print("c_valid: ", c_valid.shape)
#        print("labels: ", labels.shape)
        c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
        #AUC is only defined when there is at least one positive data.
#        print('c_label: ', c_label)
#        print('c_pred: ', c_pred)
#        print('c_valid: ', c_valid)
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
        self.MeanSquaredError = MeanSquaredError()
        print("FineTune init done")

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step2(self, model, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, prot_rep, prot_num, cellline_drp_rep, cellline_dsp_rep, is_catalyst, tasks, labels, val=False, epoch_counter=1000):
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
                loss_tmp = self.BCELoss(pred_item, label_item.float())

            if task_id in ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'ppbr_az', 'ld50_zhu', 'vdss_lombardo', 'half_life_obach', 'clearance_hepatocyte_az', 'clearance_microsome_az']:
                loss_tmp = self.L1Loss(pred_item, label_item.float())

            if task_id in ['davis', 'kiba', 'bindingdb_kd', 'gdsc1', 'gdsc2', 'oncopolypharmacology']:
                loss_tmp = self.MSELoss(pred_item, label_item.float())

            if task_id in ['buchwald-hartwig']:
                loss_tmp = self.SmoothL1Loss(pred_item, label_item.float())

            if task_id in ['esol', 'lipophilicity']:
                loss_tmp = calc_rmse(label_item.float(), pred_item)
                    
            total_loss = total_loss + loss_tmp
            tmp_loss_list[task_id].append(loss_tmp)

        assert label_start == len(labels)
                
        return total_loss/len(morgan_fp), pred, tmp_loss_list

    def train(self):
        test_loader = self.dataset.get_data_loaders(self.config['model_type'], self.config['task'])
            
        if self.config['model_type'] == "norm":
            from models.finetune_model import Mol_Encoder_norm
            encoder = Mol_Encoder_norm(hidden_dim=self.config['dim'], num_transformer_layers=self.config['layer'], num_heads=8, device=self.device)
            from models.finetune_model import finetune_model_norm_all_task_together
            model = finetune_model_norm_all_task_together(hidden_dim=self.config['dim'], tf_layer_num=self.config['layer'], num_heads=8, tasks_info=config['task_info'], encoder=encoder).to(self.device)
            pretrain_state = torch.load('/workspace/ktj/Mol-reasoning/CTRL_Code/ckpt_finetune/norm/best_weight.pth', map_location=self.device)
            missing, unexpected = model.load_state_dict(pretrain_state, strict=False)
            print("missing: ", missing)
            print("unexpected: ", unexpected)
            
            
        test_loss, test_results = self._validate(model, test_loader, epoch_counter=1000, test=True, task_name=self.config['task'])
        print(self.config['task'], ': ', test_results[self.config['task']])

    def _validate(self, model, valid_loader, epoch_counter=1000, test=False, task_name=None):
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

                if True:
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

        if test and (self.config['model_type'] == "weightintegrate" or self.config['model_type'] == "weightintegrate2"):
            for task_id, _ in weight_list.items():
                weight_list_item = torch.stack(weight_list[task_id])
                weight_list_item_mean = weight_list_item.mean(dim=0)
                print(f"weight_list_item_mean {task_id} : ", weight_list_item_mean.tolist())

        for task_id, predictions in predictions_list.items():

            if task_id != task_name:
                continue

            if task_id in ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp3a4_substrate_carbonmangels', 'herg', 'ames', 'dili', 'drugbank']:
                predictions = np.array(predictions).reshape(-1, 1)
                labels = np.array(labels_list[task_id]).reshape(-1, 1)
#                print("predictions: ", predictions)
#                print("labels: ", labels)
                AUROC = roc_auc_score(labels, predictions)
                result_list[task_id] = AUROC
                    
            if task_id == 'bbbp' or task_id == 'hiv' or task_id == 'clintox' or task_id == 'sider':
                predictions = np.array(predictions)
                pred_classes = (predictions >= 0.5).astype(int).reshape(-1)
                labels = np.array(labels_list[task_id]).astype(int).reshape(-1)
                roc_auc = accuracy_score(labels, pred_classes)
                result_list[task_id] = roc_auc

            if task_id in ['cyp2d6_veith', 'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels']:
                predictions = np.array(predictions)
                labels = np.array(labels_list[task_id])
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
    parser.add_argument('--task', type=str)
    parser.add_argument('--model', type=str, default="norm")
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    config = {}

    config['gpu'] = args.device
    config['model_type'] = args.model
    config['layer'] = args.layer
    config['dim'] = args.dim
    config['num_workers'] = args.num_workers
    config['batch_size'] = args.batch_size
    config['task'] = args.task


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

#    print("config: ", config)

    result = main(config)
