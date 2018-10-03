# predict and write result
from __future__ import absolute_import

import pandas as pd
import numpy as np

import keras
from keras.models import Sequential, Model, load_model
from keras.backend import one_hot

# vocabularies
shops = pd.read_csv('data/shops.csv')
items = pd.read_csv('data/items.csv')
item_cats = pd.read_csv('data/item_categories.csv')
SHOPS_COUNT = len(shops)
ITEMS_COUNT = len(items)
CATS_COUNT = len(item_cats)

# create test inputs
X_test = pd.read_csv('data/test.csv', dtype={'shop_id': np.int32, 'item_id': np.int32})
X_test['date_block_num'] = 34
X_test['month'] = 11

# add item price to test
data = pd.read_csv('data/train.csv', dtype={'shop_id': np.int32, 'item_it': np.int32, 'item_cnt_day':np.int32})
items = pd.read_csv('data/items.csv')
X_test['item_price'] = X_test.join(data, on='item_id', how='left', lsuffix='item_id').item_price
X_test['item_cat'] = X_test.join(items, on='item_id', how='left', lsuffix='item_id').item_category_id

# create training inputs and target
x_test = X_test.values
inputs_test = [x_test[:,i] for i in [3,1,2,4,5,6]]

model = load_model('trained_model/lr1e-05_03d11-58/weights-improvement-03-14.904227.hdf5',
	 custom_objects={'one_hot':keras.backend.one_hot,'ITEMS_COUNT':ITEMS_COUNT, 'SHOPS_COUNT':SHOPS_COUNT, 'CATS_COUNT':CATS_COUNT})

y_out = model.predict(inputs_test, verbose=1).flatten().tolist()

import csv
with open('predictions.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(('ID','item_cnt_month'))
    for i in range(len(y_out)):
        writer.writerow((i, y_out[i]))