import numpy as np
import pandas as pd
import os
import shutil
import datetime

from os.path import join as pjoin

from utils import Zscore_norm

root = '../../data/ACL18/price/preprocessed'
T, lower_bound, upper_bound, time_limit = 10, -0.005, 0.0055, '2016-09-01'
format_str = '%Y-%m-%d'
store_path = pjoin('../../data/ACL18/examples', 'T_%d' % T)

time_limit = datetime.datetime.strptime(time_limit, format_str)
f_list = os.listdir(root)

if os.path.exists(store_path):
    shutil.rmtree(store_path)
os.mkdir(store_path)
os.mkdir(pjoin(store_path, 'train'))
os.mkdir(pjoin(store_path, 'val'))

train, val = pd.DataFrame(columns=['name', 'label']), pd.DataFrame(columns=['name', 'label'])

for f in f_list:
    data = pd.read_csv(pjoin(root, f), sep='\t', header=None)
    data.columns = ['date', 'movement_percent', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    data = data.iloc[::-1]

    mp = np.asarray((data['movement_percent'])).copy()

    Zscore_norm(data, data.columns[1:])
    length = data.shape[0]

    for d in range(T, length):
        # label = data.iloc[d]['movement_percent']
        label = mp[d]
        t = datetime.datetime.strptime(data.iloc[d]['date'], format_str)

        if label >= upper_bound or label <= lower_bound:

            lag = data[d - T:d]
            filename = f.split('.')[0] + ' ' + data.iloc[d]['date']

            if (time_limit - t).days > 0:
                train = train.append({'name': filename, 'label': int(label > 0)}, ignore_index=True)
                lag.to_csv(pjoin(store_path, 'train', filename + '.csv'), index=False)
            elif (t - time_limit).days > 2 * T:
                val = val.append({'name': filename, 'label': int(label > 0)}, ignore_index=True)
                lag.to_csv(pjoin(store_path, 'val', filename + '.csv'), index=False)

train.to_csv(pjoin(store_path, 'train.csv'), index=False)
val.to_csv(pjoin(store_path, 'val.csv'), index=False)
