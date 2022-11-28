import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

def process_data(data_name):
    if data_name == 'NAB':
        data_path = './data/NAB'
        save_path = f'./process/{data_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_list = os.listdir(data_path)

        for name in data_list:
            data_dict = dict()

            df = pd.read_csv(os.path.join(data_path, name), header=None).to_numpy()
            data = df[:, 0:1].astype(float)
            label = df[:, 1]

            scaler = StandardScaler()
            lens = data.shape[0]
            train = data[:int(lens*0.5)]
            train_labels = label[:int(lens*0.5)]
            scaler.fit(train)
            train = scaler.transform(train)
            data_dict['train'] = train.tolist()
            data_dict['train_labels'] = train_labels.tolist()

            test = data[int(lens*0.5):]
            labels = label[int(lens*0.5):]
            test = scaler.transform(test)
            data_dict['test'] = test.tolist()
            data_dict['labels'] = labels.tolist()
            
            save_name = name.split('.')[0][9:]
            json.dump(data_dict, open(save_path+f'/{save_name}.json','w'))


    elif data_name[:4] == 'NASA':
        data_path = f'./data/{data_name}'
        save_path = f'./process/{data_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_list = os.listdir(data_path)
        name_list = []

        for name in data_list:
            name_pre = name.split('.')[0]
            if name_pre in name_list:
                continue
            else:
                name_list.append(name_pre)
                data_dict = dict()

                df = pd.read_csv(os.path.join(data_path, f'{name_pre}.train.out'), header=None).to_numpy()
                train = df[:, 0:1].astype(float)
                train_labels = df[:, 1]
                df = pd.read_csv(os.path.join(data_path, f'{name_pre}.test.out'), header=None).to_numpy()
                test = df[:, 0:1].astype(float)
                label = df[:, 1]

                scaler = StandardScaler()
                scaler.fit(train)
                train = scaler.transform(train)
                data_dict['train'] = train.tolist()
                data_dict['train_labels'] = train_labels.tolist()

                test = scaler.transform(test)
                data_dict['test'] = test.tolist()
                data_dict['labels'] = label.tolist()
                
                save_name = name_pre
                json.dump(data_dict, open(save_path+f'/{save_name}.json','w'))


if __name__ == '__main__':
    data_list = ['NAB', 'NASA']
    for name in data_list:
        process_data(name)
