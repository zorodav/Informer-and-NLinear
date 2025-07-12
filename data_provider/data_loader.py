import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import importlib.util
import sys
import os

import yaml
import sys
import os
from ctf4science.data_module import load_validation_dataset, _load_test_data
import importlib.util
from pathlib import Path


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Lorenz_Official(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ODE_Lorenz',
                 target='x', scale=True, timeenc=0, freq='h', train_only=False, pair_id=1):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
 
        self.pair_id = pair_id

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        train_data, validation_data, initialization_data = load_validation_dataset(self.data_path, self.pair_id, transpose=False)
        
        if self.flag == 'train':
            data = np.vstack(train_data)
            self.init_data = initialization_data if initialization_data is not None else None

            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))
                if self.init_data is not None:
                    self.init_data = np.ascontiguousarray(self.scaler.transform(self.init_data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            
        elif self.flag == 'test':
            data = _load_test_data(self.data_path, self.pair_id, transpose=False)
            self.init_data = None
            
            print("Data shape:", len(data))
            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            
        elif self.flag == 'pred':
            data = validation_data
            self.init_data = initialization_data if initialization_data is not None else None
            
            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))
                if self.init_data is not None:
                    self.init_data = np.ascontiguousarray(self.scaler.transform(self.init_data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        
        if self.flag == 'test':
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        elif self.flag == 'pred' and self.init_data is not None:
            label_portion = self.data_y[r_begin:r_begin + self.label_len]
            
            if len(self.init_data) >= self.label_len:
                init_portion = self.init_data[:self.label_len]
                
                seq_y = init_portion
                
                extended_input = np.concatenate([init_portion, seq_x], axis=0)
                seq_x = extended_input[-self.seq_len:] 
                
            else:
                seq_y = label_portion
        else:
            seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = np.zeros((seq_x.shape[0], 1))
        seq_y_mark = np.zeros((seq_y.shape[0], 1))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
            
class ODE_Lorenz(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ODE_Lorenz',
                 target='x', scale=True, timeenc=0, freq='h', train_only=True, pair_id=1):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.pair_id = pair_id

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        training_data, validation_data, initialization_data = load_validation_dataset(self.data_path, self.pair_id, transpose=False)
        print(len(training_data))
        if self.flag == 'train':
            data = np.vstack(training_data)
            print(data.shape)
            self.init_data = initialization_data if initialization_data is not None else None
            
            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                print(len(data), border1s[0], border2s[0])
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))
                if self.init_data is not None:
                    self.init_data = np.ascontiguousarray(self.scaler.transform(self.init_data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            
        elif self.flag == 'test':
            data = _load_test_data(self.data_path, self.pair_id, transpose=False)
            self.init_data = None
            
            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            
        elif self.flag == 'pred':
            data = validation_data
            self.init_data = initialization_data if initialization_data is not None else None
            
            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))
                if self.init_data is not None:
                    self.init_data = np.ascontiguousarray(self.scaler.transform(self.init_data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        
        if self.flag == 'test':
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        elif self.flag == 'pred' and self.init_data is not None:
            label_portion = self.data_y[r_begin:r_begin + self.label_len]
            
            if len(self.init_data) >= self.label_len:
                init_portion = self.init_data[:self.label_len]
                seq_y = init_portion
                
                extended_input = np.concatenate([init_portion, seq_x], axis=0)
                seq_x = extended_input[-self.seq_len:]
                
            else:
                seq_y = label_portion
        else:
            seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = np.zeros((seq_x.shape[0], 1))
        seq_y_mark = np.zeros((seq_y.shape[0], 1))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        print("Length of dataset:", len(self.data_x), self.seq_len, self.pred_len + 1, self.flag)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class PDE_KS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='PDE_KS',
                 target='u', scale=True, timeenc=0, freq='h', train_only=True, pair_id=1):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.pair_id = pair_id

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        training_data, validation_data, initialization_data = load_validation_dataset(self.data_path, self.pair_id, transpose=False)
        
        if self.flag == 'train':
            data = np.vstack(training_data)
            if len(data.shape) > 2:
                data = data.reshape(-1, data.shape[-1])
            
            self.init_data = initialization_data if initialization_data is not None else None
            if self.init_data is not None and len(self.init_data.shape) > 2:
                self.init_data = self.init_data.reshape(-1, self.init_data.shape[-1])

            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))
                if self.init_data is not None:
                    self.init_data = np.ascontiguousarray(self.scaler.transform(self.init_data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            
        elif self.flag == 'test':
            data = _load_test_data(self.data_path, self.pair_id, transpose=False)
            self.init_data = None
            
            if len(data.shape) > 2:
                data = data.reshape(-1, data.shape[-1])

            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            
        elif self.flag == 'pred':
            data = validation_data
            self.init_data = initialization_data if initialization_data is not None else None

            if len(data.shape) > 2:
                data = data.reshape(-1, data.shape[-1])
            if self.init_data is not None and len(self.init_data.shape) > 2:
                self.init_data = self.init_data.reshape(-1, self.init_data.shape[-1])

            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))
                if self.init_data is not None:
                    self.init_data = np.ascontiguousarray(self.scaler.transform(self.init_data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        
        if self.flag == 'test':
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        elif self.flag == 'pred' and self.init_data is not None:
            label_portion = self.data_y[r_begin:r_begin + self.label_len]
            
            if len(self.init_data) >= self.label_len:
                init_portion = self.init_data[:self.label_len]
                seq_y = init_portion
                
                extended_input = np.concatenate([init_portion, seq_x], axis=0)
                seq_x = extended_input[-self.seq_len:]
                
            else:
                seq_y = label_portion
        else:
            seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = np.zeros((seq_x.shape[0], 5))
        seq_y_mark = np.zeros((seq_y.shape[0], 5))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
                
class KS_Official(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='PDE_KS',
                 target='u', scale=True, timeenc=0, freq='h', train_only=True, pair_id=1):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'pred': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.pair_id = pair_id

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        training_data, validation_data, initialization_data = load_validation_dataset(self.data_path, self.pair_id, transpose=False)
        
        if self.flag == 'train':
            data = np.vstack(training_data)
            self.init_data = initialization_data if initialization_data is not None else None

            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))
                if self.init_data is not None:
                    self.init_data = np.ascontiguousarray(self.scaler.transform(self.init_data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            
        elif self.flag == 'test':
            data = _load_test_data(self.data_path, self.pair_id, transpose=False)
            self.init_data = None

            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            
        elif self.flag == 'pred':
            data = validation_data
            self.init_data = initialization_data if initialization_data is not None else None

            num_train = int(len(data) * (0.7 if not self.train_only else 1))
            num_test = int(len(data) * 0.2)
            num_vali = len(data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                self.scaler = StandardScaler()
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = np.ascontiguousarray(self.scaler.transform(data))
                if self.init_data is not None:
                    self.init_data = np.ascontiguousarray(self.scaler.transform(self.init_data))

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        
        if self.flag == 'test':
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        elif self.flag == 'pred' and self.init_data is not None:
            label_portion = self.data_y[r_begin:r_begin + self.label_len]
            
            if len(self.init_data) >= self.label_len:
                init_portion = self.init_data[:self.label_len]
                seq_y = init_portion
                
                extended_input = np.concatenate([init_portion, seq_x], axis=0)
                seq_x = extended_input[-self.seq_len:]
                
            else:
                seq_y = label_portion
        else:
            seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = np.zeros((seq_x.shape[0], 5))
        seq_y_mark = np.zeros((seq_y.shape[0], 5))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)