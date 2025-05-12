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
                            outputs = self.model(batch_x)
                        else:
                            if self.args['output_attention']:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
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
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
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
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                else:
                    if 'Linear' in self.args['model']['name']:
                            outputs = self.model(batch_x)
                    else:
                        if self.args['output_attention']:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args['features'] == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args['pred_len']:, f_dim:]
                batch_y = batch_y[:, -self.args['pred_len']:, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args['test_flop']:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
            
        # Concatenate predictions and ground truth arrays
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

                # Reshape if arrays are 3D: convert from (batch, pred_len, features) to (features, batch * pred_len)
        if self.args['dataset']['name'] == 'ODE_Lorenz':
            if preds.ndim == 3:
                preds = preds.transpose(2, 0, 1).reshape(preds.shape[2], -1)
            if trues.ndim == 3:
                trues = trues.transpose(2, 0, 1).reshape(trues.shape[2], -1)
        
            if self.args['model']['name'] == 'NLinear':
                preds = preds[:, :2000]
                trues = trues[:, :2000]
        else:
            if self.args['dataset']['name'] == 'PDE_KS':
                if preds.ndim == 3:
                    preds = preds.transpose(2, 0, 1).reshape(preds.shape[2], -1)
                if trues.ndim == 3:
                    trues = trues.transpose(2, 0, 1).reshape(trues.shape[2], -1)
                if trues.shape[0] != preds.shape[0]:
                    min_features = min(trues.shape[0], preds.shape[0])
                    trues = trues[:min_features, :]
                    preds = preds[:min_features, :]


        eval_results = evaluate_custom(
            dataset_name=self.args['dataset']['name'],
            pair_id=self.args['dataset']['pair_id'],
            truth=trues,
            prediction=preds,
            metrics=self.args.get('evaluation_metrics', None)
        )
        print("Evaluation results:", eval_results)

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.savetxt(folder_path + 'predicted.txt', preds)
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

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args['pred_len'], batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args['label_len'], :], dec_inp], dim=1).float()
                # encoder-decoder
                if self.args['use_amp']:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args['model']:
                            outputs = self.model(batch_x)
                        else:
                            if self.args['output_attention']:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args['model']:
                        outputs = self.model(batch_x)
                    else:
                        if self.args['output_attention']:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        # After concatenating, squeeze out the extra dimension; now shape becomes (177, 24, 3)
        preds = np.squeeze(preds, axis=1)  # current shape: (177, 24, 3)

        # Insert the code to select a single prediction window if necessary:
        if preds.shape[0] > 1:
            # If predictions is (177, T, D) but you need a single window,
            # select the first prediction and transpose if necessary.
            preds = preds[0]      # shape becomes (24, 3)
            preds = preds.T       # shape becomes (3, 24), which should match the truth's shape

        # Now the predictions are 2D. We then set up for inverse transform:
        N, T = preds.shape    # N is number of features, T is number of time steps
        preds_2d = preds      # Already 2D, so no further reshaping is needed

        # Apply inverse transform on 2D array using the scaler in pred_data:
        preds_transformed = pred_data.scaler.inverse_transform(preds_2d)

        # If needed, you can reshape preds_transformed back to the expected shape.
        # For example, if the evaluation expects a 3D array, you may do:
        # preds = preds_transformed.reshape(1, T, N)
        # Otherwise, if the evaluation expects (D, T) simply assign:
        preds = preds_transformed        # --- Use the eval_module here ---
        # We assume eval_config takes (dataset, predictions) and returns evaluation metrics.
        # filepath: ...\exp_main.py (inside your predict or test method)

        eval_results = evaluate(
            dataset_name=self.args['dataset']['name'],
            pair_id=self.args['dataset']['pair_id'],
            prediction=preds,
            metrics=self.args.get('evaluation_metrics', None)
        )
        print("Evaluation results:", eval_results)
        # ----------------------------------

        # Save the results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.savetxt(os.path.join(folder_path, 'real_prediction.txt'), preds, fmt='%.6f')
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1),
                    columns=pred_data.cols).to_csv(os.path.join(folder_path, 'real_prediction.csv'), index=False)

        return