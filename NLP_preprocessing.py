from __future__ import absolute_import
import pandas as pd
import numpy as np

import polyglot
from polyglot.detect import Detector
from polyglot.mapping import Embedding
import string

data = pd.read_csv('data/train.csv', dtype={'shop_id': np.int32, 'item_id': np.int32, 'item_cnt_day':np.int32})

# vocabularies
shops = pd.read_csv('data/shops.csv')
items = pd.read_csv('data/items.csv')
item_cats = pd.read_csv('data/item_categories.csv')
SHOPS_COUNT = len(shops)
ITEMS_COUNT = len(items)
CATS_COUNT = len(item_cats)


VOCAB_SIZE = 64
embeddings_ru = Embedding.load("data/ru_embeddings_pkl.tar.bz2")
embeddings_en = Embedding.load("data/en_embeddings_pkl.tar.bz2")
punctuation_table = str.maketrans({key: None for key in string.punctuation+string.digits})

def encoder(entries):
    encoded = []
    for i,entry in enumerate(entries.tolist()):
        entry = entry.translate(punctuation_table)

        temp = []
        for word in entry.split(" "):
            if word.replace(" ", "") in embeddings_en:
                temp.append(embeddings_en[word])
            elif word.replace(" ", "") in embeddings_ru:
                temp.append(embeddings_ru[word]) 
            else:
                temp.append(np.array([0]*64)) 
        temp = np.array(temp).mean(axis=0)
        encoded.append(temp)
    return encoded

shop_vec = encoder(shops.shop_name)
item_vec = encoder(items.item_name)
cat_vec = encoder(item_cats.item_category_name)

shops['shop_vec'] = shop_vec
items['item_vec'] = item_vec
item_cats['cat_vec'] = cat_vec

def preprocessing(dt):
    # add feature month to train data
    dt['month'] = dt.date_block_num % 12
    dt['item_category_id'] = dt.join(items, on='item_id', how='left', lsuffix='item_id').item_category_id
    dt['item_vec'] = dt.join(items, on='item_id', how='left', rsuffix='ref').item_vec
    dt['cat_vec'] = dt.join(item_cats, on='item_category_id', how='left', rsuffix='ref').cat_vec
    dt['shop_vec'] = dt.join(shops, on='shop_id', how='left', rsuffix='ref').shop_vec
    return dt


X = pd.DataFrame(data.groupby(['date_block_num','shop_id', 'item_id', 'item_price'])['item_cnt_day'].sum()).reset_index()
X = preprocessing(X)
X = X[['date_block_num','month','item_vec','cat_vec','shop_vec','item_cnt_day']]

X_test = pd.read_csv('data/test.csv', dtype={'shop_id': np.int32, 'item_id': np.int32})
X_test['date_block_num'] = 34
X_test['month'] = 11

X_test = preprocessing(X_test)
X_test = X_test[['date_block_num','month','item_vec','cat_vec','shop_vec']]

X_test = preprocessing(X_test)
X_test = X_test[['date_block_num','month','item_vec','cat_vec','shop_vec']]

X.to_csv('X.csv')
X_test.to_csv('X_test.csv')