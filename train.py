from __future__ import absolute_import

import pandas as pd
import numpy as np
from datetime import datetime

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Input, Concatenate, Flatten, BatchNormalization, Activation, Dropout, Lambda
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,TerminateOnNaN
from keras import optimizers, initializers
from keras.backend import one_hot, sqrt
from keras.losses import mean_squared_error

# load original training data
data = pd.read_csv('data/train.csv', dtype={'shop_id': np.int32, 'item_id': np.int32, 'item_cnt_day':np.int32})

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


# create training inputs and target
x = X.values
inputs = [x[:,i] for i in [0,1,2,5,3,6]]
y = x[:,4]

# training spec
keras.backend.clear_session()
NUM_EPOCHS = 50
LEARNING_RATE= 0.00001
BETA1=0.90
# shops: 60 item_num: 22170 item_cat: 84
SHOP_EMB_DIM, ITEM_EMB_DIM, CAT_EMB_DIM = (16,128,16)

def build_model():
#   input layers--numeric 
    date = Input(shape=(1,), name='date_input')
    price = Input(shape=(1,), name='price_input')
#   input layers--categorical
    shop = Input(shape=(1,), name='shop_input',dtype='int32')
    item = Input(shape=(1,), name='item_input', dtype='int32')
    month = Input(shape=(1,), name='month_input', dtype='int32')
    cat = Input(shape=(1,), name='category_input', dtype='int32')
    
   
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
    preds = Dropout(0.1)(preds)
    preds = Dense(16, activation='relu',name='dense2')(preds)
    preds = Dropout(0.1)(preds)
    preds = Dense(16, activation='relu', name='dense3')(preds)

    oh_shop = Lambda(lambda x: one_hot(x, SHOPS_COUNT), output_shape = (1, SHOPS_COUNT))(shop)
    oh_cat = Lambda(lambda x: one_hot(x, CATS_COUNT),output_shape = (1, CATS_COUNT))(cat)
    oh_item = Lambda(lambda x: one_hot(x, ITEMS_COUNT),output_shape = (1, ITEMS_COUNT))(item)

    wide_shop = Dense(1, name='wide_shop')(oh_shop)
    wide_cat = Dense(1, name='wide_cat')(oh_cat)
    wide_item = Dense(4, name='wide_item')(oh_item)    

    wide_shop_flat = Flatten(name='wide_shop_flat')(wide_shop)
    wide_cat_flat = Flatten(name='wide_cat_flat')(wide_cat)
    wide_item_flat = Flatten(name='wide_item_flat')(wide_item)
    wide_item_flat = BatchNormalization(name='wide_item_batchnorm')(wide_item_flat)

    all_inputs = Concatenate(axis=-1, name='all_inputs_concat')([wide_item_flat, wide_shop_flat, wide_cat_flat, preds])
    all_inputs = BatchNormalization(name='all_inputs_batchnorm')(all_inputs)


    # preds = Dense(8,activation ='relu', name='out_nn')(all_inputs)
    preds = Dense(1, activation='relu', name='final_out')(all_inputs)

    return Model(inputs=[date, shop, item, month, price, cat], outputs=preds)
    

model = build_model()
model.summary()

adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA1)

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))
   
model.compile(optimizer = adam,loss='mean_squared_error', metrics=[rmse])

OUTPUT_DIR = './trained_model/'+ 'lr' + str(LEARNING_RATE) + '_' + datetime.now().strftime("%dd%H-%M")
filepath = OUTPUT_DIR +'/' + "weights-improvement-{epoch:02d}-{val_rmse:.6f}.hdf5"

# model = load_model('keras/weights-improvement-02-14.970410.hdf5')
# model.load_weights('trained_model/lr0.0001_04d16-09/weights-improvement-08-1.191472.hdf5')

callbacks = [
             TerminateOnNaN(),
             ModelCheckpoint(filepath=filepath, monitor='val_rmse', verbose=1, period=1, save_best_only=True),
             # EarlyStopping(patience=2, monitor='loss'),
             TensorBoard(log_dir=OUTPUT_DIR, write_images=False, histogram_freq=1, write_grads=True),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
             keras.callbacks.CSVLogger('log.csv', separator=',', append=False)
]

model.fit(inputs, y, batch_size = 1024, epochs=NUM_EPOCHS, callbacks=callbacks, shuffle=True,
          validation_split=0.01)
