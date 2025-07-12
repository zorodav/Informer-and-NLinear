from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time
from ctf4science.eval_module import evaluate_custom, evaluate
import warnings
import matplotlib.pyplot as plt
import numpy as np
from models.Autoformer import Model as Autoformer
from models.Transformer import Model as Transformer
from models.Informer import Model as Informer
from models.DLinear import Model as DLinear
from models.Linear import Model as Linear
from models.NLinear import Model as NLinear
import importlib.util


import sys
from pathlib import Path

# Go up four levels to reach the project root
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args['model']['name']]
        model = model(self.args)
        if self.args['use_multi_gpu'] & self.args['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.args['device_ids'])
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args['pred_len']:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float()
                # encoder - decoder
                if self.args['use_amp']:
                    with torch.amp.autocast('cuda'):
                        if 'Linear' in self.args['model']['name']:
                            outputs = self.model(batch_x, autoregressive=True)
                        else:
                            if self.args['output_attention']:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, autoregressive=True)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, autoregressive=True)
                else:
                    if 'Linear' in self.args['model']['name']:
                        outputs = self.model(batch_x, autoregressive=True)
                    else:
                        if self.args['output_attention']:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, autoregressive=True)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, autoregressive=True)
                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, -self.args['pred_len']:, f_dim:]
                batch_y = batch_y[:, -self.args['pred_len']:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        if not self.args["training"]["train_only"]:
            vali_data, vali_loader = self._get_data(flag='pred')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args['checkpoints'], setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args['patience'], verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args['use_amp']:
            scaler = torch.amp.GradScaler('cuda')

        for epoch in range(self.args['train_epochs']):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                else:
                    raise ValueError(f"Expected 4 elements in batch, got {len(batch)}")
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float()

                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args['pred_len']:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float()

                # encoder - decoder
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args['model']['name']:
                            outputs = self.model(batch_x)
                        else:
                            if self.args['output_attention']:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args['features'] == 'MS' else 0
                        outputs = outputs[:, -self.args['pred_len']:, f_dim:]
                        batch_y = batch_y[:, -self.args['pred_len']:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args['model']['name']:
                            outputs = self.model(batch_x)
                    else:
                        if self.args['output_attention']:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]       
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args['features'] == 'MS' else 0
                    outputs = outputs[:, -self.args['pred_len']:, f_dim:]
                    batch_y = batch_y[:, -self.args['pred_len']:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args['train_epochs'] - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args['use_amp']:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args['training']['train_only']:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        pred_data, pred_loader = self._get_data(flag='pred') 
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        next_step_preds = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()

                if hasattr(batch_y, 'shape') and batch_y.shape[1] >= self.args['label_len']:
                    init_context = batch_y[:, :self.args['label_len'], :]
                    dec_inp = torch.zeros([batch_x.shape[0], 1, batch_x.shape[2]]).float()
                    dec_inp = torch.cat([init_context, dec_inp], dim=1).float()
                    dec_mark = batch_y_mark[:, :self.args['label_len'] + 1, :]
                    
                    if 'Linear' in self.args['model']['name']:
                        full_input = torch.cat([batch_x, init_context], dim=1)
                        input_seq = full_input[:, -self.args['seq_len']:, :]
                    else:
                        input_seq = batch_x
                else:
                    dec_inp = torch.zeros([batch_x.shape[0], 1, batch_x.shape[2]]).float()
                    dec_inp = torch.cat([batch_x[:, -self.args['label_len']:, :], dec_inp], dim=1).float()
                    dec_mark = batch_x_mark[:, -self.args['label_len']:, :]
                    input_seq = batch_x
                
                if 'Linear' in self.args['model']['name']:
                    if self.args['use_amp']:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(input_seq, autoregressive=False)
                    else:
                        outputs = self.model(input_seq, autoregressive=False)
                    next_step_pred = outputs[:, 0, :]
                else:
                    if self.args['use_amp']:
                        with torch.cuda.amp.autocast():
                            if self.args['output_attention']:
                                outputs = self.model(input_seq, batch_x_mark, dec_inp, dec_mark, autoregressive=False)[0]
                            else:
                                outputs = self.model(input_seq, batch_x_mark, dec_inp, dec_mark, autoregressive=False)
                    else:
                        if self.args['output_attention']:
                            outputs = self.model(input_seq, batch_x_mark, dec_inp, dec_mark, autoregressive=False)[0]
                        else:
                            outputs = self.model(input_seq, batch_x_mark, dec_inp, dec_mark, autoregressive=False)
                    next_step_pred = outputs[:, -1, :]

                f_dim = -1 if self.args['features'] == 'MS' else 0
                pred = next_step_pred[:, f_dim:].detach().cpu().numpy()
                next_step_preds.append(pred)

        preds = np.concatenate(next_step_preds, axis=0)
        
        if hasattr(pred_data, 'scaler') and pred_data.scaler is not None:
            preds = pred_data.scaler.inverse_transform(preds)

        if self.args['dataset']['name'] in ['KS_Official', 'Lorenz_Official', 'ODE_Lorenz', 'PDE_KS']:
            if preds.shape[0] > 1:
                preds = preds[0]
                preds = preds.T

        eval_results = evaluate(
            dataset_name=self.args['dataset']['name'],
            pair_id=self.args['dataset']['pair_id'],
            prediction=preds,
            metrics=self.args.get('evaluation_metrics', None)
        )
        print("Evaluation results:", eval_results)
        
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args['checkpoints'], setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()

                dec_inp = torch.zeros([batch_y.shape[0], self.args['pred_len'], batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float()
                
                if 'Linear' in self.args['model']['name']:
                    full_context = torch.cat([batch_x, batch_y[:, :self.args['label_len'], :]], dim=1)
                    input_seq = full_context[:, -self.args['seq_len']:, :]
                else:
                    input_seq = batch_x
                
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args['model']['name']:
                            outputs = self.model(input_seq, autoregressive=True)
                        else:
                            if self.args['output_attention']:
                                outputs = self.model(input_seq, batch_x_mark, dec_inp, batch_y_mark, autoregressive=True)[0]
                            else:
                                outputs = self.model(input_seq, batch_x_mark, dec_inp, batch_y_mark, autoregressive=True)
                else:
                    if 'Linear' in self.args['model']['name']:
                        outputs = self.model(input_seq, autoregressive=True)  
                    else:
                        if self.args['output_attention']:
                            outputs = self.model(input_seq, batch_x_mark, dec_inp, batch_y_mark, autoregressive=True)[0]
                        else:
                            outputs = self.model(input_seq, batch_x_mark, dec_inp, batch_y_mark, autoregressive=True)
                
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.squeeze(preds, axis=1)

        if preds.shape[0] > 1:
            preds = preds[0]
            preds = preds.T

        N, T = preds.shape
        preds_2d = preds
        preds_transformed = pred_data.scaler.inverse_transform(preds_2d)
        preds = preds_transformed

        eval_results = evaluate(
            dataset_name=self.args['dataset']['name'],
            pair_id=self.args['dataset']['pair_id'],
            prediction=preds,
            metrics=self.args.get('evaluation_metrics', None)
        )
        print("Evaluation results:", eval_results)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savetxt(os.path.join(folder_path, 'real_prediction.txt'), preds, fmt='%.6f')
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1),
                    columns=pred_data.cols).to_csv(os.path.join(folder_path, 'real_prediction.csv'), index=False)

        return 