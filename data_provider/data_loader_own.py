from cmath import isnan
import os
import numpy as np
import pandas as pd
import os
import json
import torch
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def read_data(data_name, root_path, flag, idx):

    data_list = os.listdir(f'./process/{data_name}')
    data_list.sort()
    name = data_list[idx]
    data = json.load(open(f'./process/{data_name}/{name}'))
    train = np.array(data['train'])
    try:
        train_y = np.array(data['train_labels']).reshape(-1)
    except:
        train_y = 0
    test = np.array(data['test'])
    labels = np.array(data['labels']).reshape(-1)
    
    train = np.nan_to_num(train)
    test = np.nan_to_num(test)
    y = labels

    return train, train_y, test, y


class SLoadData(Dataset):
    def __init__(self, root_path, seq_len, data_name, flag='train',
                    task='imp', features='S', idx=1):
        # size [seq_len, label_len, pred_len]
        # info
        # init
        assert flag in ['train', 'test', 'vali']

        self.scaler = StandardScaler()
        self.features = features
        self.task = task
        self.flag = flag
        self.idx = idx
        self.root_path = root_path
        self.data_name = data_name
        self.scaler = StandardScaler()
        self.seq_len = seq_len
        if self.flag == 'train':
            self.step = 1
        else:
            self.step = self.seq_len
        self.__read_data__()

    def __read_data__(self):
        self.train, train_data_y, self.test, labels = read_data(self.data_name, self.root_path, 'train', self.idx)
        self.labels = labels[:, np.newaxis]
        self.train_data_y = train_data_y[:, np.newaxis]

    def __getitem__(self, index):

        index = index * self.step
        if self.flag == "train":
            train_data_y = self.train_data_y[index:index + self.seq_len]
            train_data_y[train_data_y==0] = -1
            return np.float32(self.train[index:index + self.seq_len]), np.float32(train_data_y)
        elif (self.flag == 'test'):
            test_data_y = self.labels[index:index + self.seq_len]
            test_data_y[test_data_y==0] = -1
            return np.float32(self.test[index:index + self.seq_len]), np.float32(test_data_y)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.seq_len) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.seq_len) // self.step + 1
        # return 32


class ALoadData(Dataset):
    def __init__(self, root_path, seq_len, data_name, flag='train',
                    task='imp', features='S', scale=True, idx=1):
        # size [seq_len, label_len, pred_len]
        # info
        # init
        assert flag in ['train', 'test', 'vali']

        self.scaler = StandardScaler()
        self.features = features
        self.task = task
        self.idx = idx
        self.flag = flag
        self.root_path = root_path
        self.data_name = data_name
        self.scaler = StandardScaler()
        self.seq_len = seq_len
        if self.flag == 'train':
            self.step = 1
            # self.step = self.seq_len
        else:
            self.step = self.seq_len
        self.__read_data__()

    def __read_data__(self):
        self.train, _, self.test, self.labels = read_data(self.data_name, self.root_path, 'train', self.idx)
        self.labels = self.labels[:, np.newaxis]

    def __getitem__(self, index):

        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.seq_len])

        elif (self.flag == 'test'):
            test_data_y = self.labels[index:index + self.seq_len]
            test_data_y[test_data_y==0] = -1
            return np.float32(self.test[index:index + self.seq_len]), np.float32(test_data_y)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.seq_len) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.seq_len) // self.step + 1



class CLoadData(Dataset):
    def __init__(self, root_path, seq_len, data_name, flag='train',
                    task='imp', features='S', scale=True, idx=1):
        # size [seq_len, label_len, pred_len]
        # info
        # init
        assert flag in ['train', 'test', 'vali']
        self.idx = idx
        self.scaler = StandardScaler()
        self.features = features
        self.task = task
        self.flag = flag
        self.root_path = root_path
        self.data_name = data_name
        self.scaler = StandardScaler()
        self.seq_len = seq_len
        if self.flag == 'train':
            self.step = 1
            # self.step = self.seq_len
        else:
            self.step = self.seq_len
        self.__read_data__()

    def __read_data__(self):
        self.train, _, self.test, self.labels = read_data(self.data_name, self.root_path, 'train', self.idx)
        self.labels = self.labels[:, np.newaxis]

    def __getitem__(self, index):

        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.seq_len])

        elif (self.flag == 'test'):
            test_data_y = self.labels[index:index + self.seq_len]
            test_data_y[test_data_y==0] = -1
            return np.float32(self.test[index:index + self.seq_len]), np.float32(test_data_y)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.seq_len) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.seq_len) // self.step + 1
        # return 32

