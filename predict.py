# predict and write result
from __future__ import absolute_import
import pandas as pd
import numpy as np

from keras.models import Sequential, Model, load_model

# create test inputs
X_test = pd.read_csv('data/test.csv')
X_test['date_block_num'] = 34
X_test['month'] = 11

# add item price to test
data = pd.read_csv('data/train.csv')
items = pd.read_csv('data/items.csv')
X_test['item_price'] = X_test.join(data, on='item_id', how='left', lsuffix='item_id').item_price
X_test['item_cat'] = X_test.join(items, on='item_id', how='left', lsuffix='item_id').item_category_id

# create training inputs and target
x_test = X_test.values
inputs_test = [x_test[:,i] for i in [3,1,2,4,5,6]]

model = load_model('trained_model/lr0.001_02d11-17/weights-improvement-01-15.106428.hdf5')
y_out = model.predict(inputs_test, verbose=1).flatten().tolist()

import csv
with open('predictions.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(('ID','item_cnt_month'))
    for i in range(len(y_out)):
        writer.writerow((i, y_out[i]))