#import torch
#torch.autograd.set_detect_anomaly(True)
import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from dataset.CTRL_pretrain_dataset import MoleculeDatasetWrapper
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class pretrain_stage(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        
        self.dataset = dataset

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, dist_list, idx_list, edge_attr_list, edge_index_list, smiles_list):
    
        total_loss, loss_dict = model(data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, dist_list, idx_list, edge_attr_list, edge_index_list)
        
        return total_loss, loss_dict

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders(self.config['model_type'])
            
        if self.config['model_type'] == "norm":
            from models.pretrain_model import Mol_Encoder_norm
            encoder = Mol_Encoder_norm(hidden_dim=self.config['dim'], num_transformer_layers=self.config['layer'], num_heads=8, dropout=0.5).to(self.device)
            from models.pretrain_model import pretrain_model_norm
            model = pretrain_model_norm(encoder=encoder, batch_size=self.config['batch_size']).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=self.config['weight_decay']
        )

        model_checkpoints_folder = f"./ckpt_pretrain/{self.config['model_type']}/"        

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print_loss_dict_bn = self.config['print_loss_dict_bn']
        print_time_bn = self.config['print_time_bn']
        run_val_bn = self.config['run_val_bn']

        for epoch_counter in range(self.config['epochs']):
            import time
            start_time = time.time()
            total_loss = 0.0
            counter = 0
            
            all_loss_dict = {'desc_int': [], 'desc_float': [], 'fp_morgan': [], 'fp_maccs': [], 'fp_daylight': [], 'an': [], 'chi': [], 'aro': [], 'hyb': [], 'deg': [], 'hs': [], 'cha': [], 'iir': [], 'ele': [], 'bt': [], 'bd': [], 'bir': [], 'bc': [], 'bs': [], 'bl': [], "cll": [], "featsplit": [], "sp": [], "kl": []}
            for bn, (data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, dist_list, idx_list, edge_attr_list, edge_index_list, smiles_list) in enumerate(train_loader):
                if self.config['model_type'] == "importrec":
                    valid_n_iter = 2
                if bn % print_time_bn == 0:
                    print("---", "bn: ", bn, "---")
                start_time = time.time()
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
                dist_list = dist_list.to(self.device)
                idx_list = idx_list.to(self.device)
                edge_attr_list = edge_attr_list.to(self.device)
                edge_index_list = edge_index_list.to(self.device)

                loss, loss_dict = self._step(model, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, dist_list, idx_list, edge_attr_list, edge_index_list, smiles_list)

                for loss_key in all_loss_dict.keys():
                    if loss_key in loss_dict:
                        all_loss_dict[loss_key].append(loss_dict[loss_key].item())
                
                loss.backward()
                optimizer.step()
                
                n_iter += 1

                if (bn + 1) % print_loss_dict_bn == 0:
                    for loss_key, loss_value in all_loss_dict.items():
                        if len(loss_value) != 0:
                            print(loss_key, (sum(loss_value) / len(loss_value)))
                    all_loss_dict = {'desc_int': [], 'desc_float': [], 'fp_morgan': [], 'fp_maccs': [], 'fp_daylight': [], 'an': [], 'chi': [], 'aro': [], 'hyb': [], 'deg': [], 'hs': [], 'cha': [], 'iir': [], 'ele': [], 'bt': [], 'bd': [], 'bir': [], 'bc': [], 'bs': [], 'bl': [], "cll": [], "featsplit": [], "sp": [], "kl": []}

                # validate the model if requested
                if (bn + 1) % run_val_bn == 0:
                    valid_loss, valid_loss_dict = self._validate(model, valid_loader, print_time_bn)
                    print(epoch_counter, bn, valid_loss, '(validation)')
                    # save the model weights
                    best_valid_loss = valid_loss
                    print("save in ", os.path.join(model_checkpoints_folder, f'model_{epoch_counter}_{valid_n_iter}_weight.pth'))
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, f'model_{epoch_counter}_{valid_n_iter}_weight.pth'))
                    print("valid_loss: ", valid_loss)
                    for loss_key, loss_value in valid_loss_dict.items():
                        if len(loss_value) != 0:
                            print(loss_key, (sum(loss_value) / len(loss_value)))
            
                    valid_n_iter += 1
            
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

    def _validate(self, model, valid_loader, print_time_bn):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0

            all_loss_dict = {'desc_int': [], 'desc_float': [], 'fp_morgan': [], 'fp_maccs': [], 'fp_daylight': [], 'an': [], 'chi': [], 'aro': [], 'hyb': [], 'deg': [], 'hs': [], 'cha': [], 'iir': [], 'ele': [], 'bt': [], 'bd': [], 'bir': [], 'bc': [], 'bs': [], 'bl': [], "cll": [], "featsplit": [], "sp": [], "kl": []}
        
            start_time = time.time()
            for bn, (data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, dist_list, idx_list, edge_attr_list, edge_index_list, smiles_list) in enumerate(valid_loader):
                if bn % print_time_bn == 0:
                    print("---", "bn: ", bn, "---")
                    print("time 0: ", time.time() - start_time)
                start_time = time.time()
                data_2D = data_2D.to(self.device)
                data_3D1 = data_3D1.to(self.device)
                data_3D2 = data_3D2.to(self.device)
                morgan_fp = morgan_fp.to(self.device)
                maccs_fp = maccs_fp.to(self.device)
                daylight_fp = daylight_fp.to(self.device)
                descriptor_int = descriptor_int.to(self.device)
                descriptor_float = descriptor_float.to(self.device)
                src_mask = src_mask.to(self.device)
                dist_list = dist_list.to(self.device)
                idx_list = idx_list.to(self.device)
                edge_attr_list = edge_attr_list.to(self.device)
                edge_index_list = edge_index_list.to(self.device)

                loss, loss_dict = self._step(model, data_2D, data_3D1, data_3D2, morgan_fp, maccs_fp, daylight_fp, descriptor_int, descriptor_float, src_mask, mask_3d, dist_list, idx_list, edge_attr_list, edge_index_list, smiles_list, bn)
                for loss_key, loss_value in all_loss_dict.items():
                    if loss_key not in loss_dict.keys():
                        continue
                    all_loss_dict[loss_key].append(loss_dict[loss_key].item())
                valid_loss += float(loss.item())
                counter += 1
            for loss_key, loss_value in all_loss_dict.items():
                if len(all_loss_dict[loss_key]) != 0:
                    print(loss_key, sum(loss_value)/len(loss_value))
        
        model.train()
        print("valid_loss: ", valid_loss)
        return valid_loss, all_loss_dict


def main(args):
    config = {}
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['init_lr'] = args.init_lr
    config['weight_decay'] = args.weight_decay
    config['gpu'] = args.device
    config['model_type'] = args.model
    config['layer'] = args.layer
    config['dim'] = args.dim
    config['print_loss_dict_bn'] = args.print_loss_dict_bn
    config['print_time_bn'] = args.print_time_bn
    config['run_val_bn'] = args.run_val_bn
    config['num_workers'] = args.num_workers
    config['save_every_n_epochs'] = args.save_every_n_epochs

    print("config: ", config)

    dataset = MoleculeDatasetWrapper(config['batch_size'], config['num_workers'])
    pretrain_model = pretrain_stage(dataset, config)
    pretrain_model.train()


import argparse
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    parser.add_argument('--model', type=str, default="norm")
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--init_lr', type=float, default=0.0004)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--print_loss_dict_bn', type=int, default=50)
    parser.add_argument('--print_time_bn', type=int, default=50)
    parser.add_argument('--run_val_bn', type=int, default=30000)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    args = parser.parse_args()
    main(args)
