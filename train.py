from __future__ import absolute_import

import pandas as pd
import numpy as np
from datetime import datetime

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Input, Concatenate, Flatten, BatchNormalization, Activation, Dropout, Lambda
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,TerminateOnNaN
from keras import optimizers, initializers

# load original training data
data = pd.read_csv('data/train.csv')

# training data aggregation by day_block
X = pd.DataFrame(data.groupby(['date_block_num','shop_id', 'item_id', 'item_price'])['item_cnt_day'].sum()).reset_index()

# vocabularies
shops = pd.read_csv('data/shops.csv')
items = pd.read_csv('data/items.csv')
item_cats = pd.read_csv('data/item_categories.csv')
SHOPS_COUNT = len(shops)
ITEMS_COUNT = len(items)
CATS_COUNT = len(item_cats)

# add feature month to train data
X['month'] = X.date_block_num % 12
# add item categories to train data
X['item_cat'] = X.join(items, on='item_id', how='left', lsuffix='item_id').item_category_id

# test data preparation
# X_test = pd.read_csv('data/test.csv')
# X_test['date_block_num'] = 34
# X_test['month'] = 11

# # add item price to test
# X_test['item_price'] = X_test.join(data, on='item_id', how='left', lsuffix='item_id').item_price
# X_test['item_cat'] = X_test.join(items, on='item_id', how='left', lsuffix='item_id').item_category_id
# # # create test inputs
# x_test = X_test.values
# inputs_test = [x_test[:,i] for i in [3,1,2,4,5,6]]

# create training inputs and target
x = X.values
inputs = [x[:,i] for i in [0,1,2,5,3,6]]
y = x[:,4]

# training spec
keras.backend.clear_session()
NUM_EPOCHS = 10
LEARNING_RATE= 0.001
# shops: 60 item_num: 22170 item_cat: 84
SHOP_EMB_DIM, ITEM_EMB_DIM, CAT_EMB_DIM = (16,128,16)

def build_model():
#   input layers--numeric 
    date = Input(shape=(1,), name='date_input')
    price = Input(shape=(1,), name='price_input')
#   input layers--categorical
    shop = Input(shape=(1,), name='shop_input')
    item = Input(shape=(1,), name='item_input')
    month = Input(shape=(1,), name='month_input')
    cat = Input(shape=(1,), name='category_input')
    
#     weight_init = initializers.RandomNormal(mean=1, stddev=2)
#     bias_init = initializers.RandomNormal(mean=0, stddev=0.5)
   
    shop_emb = Embedding(input_dim=SHOPS_COUNT, output_dim=SHOP_EMB_DIM, input_length=1, name='shop_emb')(shop)
    shop_emb = Flatten(name='shop_flatten')(shop_emb)
    shop_emb = BatchNormalization(name='shop_batchnorm')(shop_emb)

    month_emb = Embedding(input_dim=12, output_dim=1, input_length=1, name='month_emb')(month)
    month_emb = Flatten(name='month_flat')(month_emb)

    item_emb = Embedding(input_dim=ITEMS_COUNT, output_dim=ITEM_EMB_DIM,input_length=1, name='item_emb')(item) 
    item_emb = Flatten(name='item_flatten')(item_emb)
    item_emb = BatchNormalization(name='item_batchnorm')(item_emb)
    
    cat_emb = Embedding(input_dim=CATS_COUNT, output_dim=CAT_EMB_DIM, input_length=1, name='cat_emb')(cat)
    cat_emb = Flatten(name='cat_flatten')(cat_emb)
    cat_emb = BatchNormalization(name='cat_batchnorm')(cat_emb)

    inputs = Concatenate(axis=-1, name='inputs_concat')([date, shop_emb, item_emb, month_emb, price, cat_emb])
    inputs_batch = BatchNormalization(name='inputs_batchnorm')(inputs)
    
    preds = Dense(48, activation='relu', name='dense1')(inputs_batch)
    # preds = Dropout(0.1)(preds)
    preds = Dense(16, activation='relu',name='dense2')(preds)
    # preds = Dropout(0.1)(preds)
    preds = Dense(16, activation='relu', name='dense3')(preds)
    preds = Dense(1,activation ='softplus', name='output')(preds)
    return Model(inputs=[date, shop, item, month, price, cat], outputs=preds)
    

model = build_model()
model.summary()

# model.load_weights('./keras/weights-improvement-01-16.679170.hdf5')
adam = optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer = adam,loss='mean_squared_error')
# model.save(OUTPUT_DIR)

OUTPUT_DIR = './trained_model/'+ 'lr' + str(LEARNING_RATE) + '_' + datetime.now().strftime("%dd%H-%M")
filepath = OUTPUT_DIR +'/' + "weights-improvement-{epoch:02d}-{val_loss:.6f}.hdf5"

# model = load_model('keras/weights-improvement-02-14.970410.hdf5')

callbacks = [
             TerminateOnNaN(),
             ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, period=1, save_best_only=True),
             EarlyStopping(patience=3, monitor='loss'),
             TensorBoard(log_dir=OUTPUT_DIR, write_images=True, histogram_freq=1, write_grads=True),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
             keras.callbacks.CSVLogger('log.csv', separator=',', append=False)
]

model.fit(inputs, y, batch_size = 128, epochs=NUM_EPOCHS, callbacks=callbacks, shuffle=True,
          validation_split=0.01)
