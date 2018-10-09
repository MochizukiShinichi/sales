from __future__ import absolute_import
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Input, Concatenate, Flatten, BatchNormalization, Activation, Dropout, Lambda
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,TerminateOnNaN
from keras import optimizers, initializers
from keras.backend import sqrt
from keras.losses import mean_squared_error
from datetime import datetime


# training data aggregation by day_block
X = pd.read_csv('X.csv')

# create training inputs and target
x = X.values
inputs = [x[:,i].tolist() for i in range(x.shape[1]-1)]
y = x[:,-1]

# training spec
keras.backend.clear_session()
NUM_EPOCHS = 50
LEARNING_RATE= 0.001
BETA1=0.90

def build_model():
#     features: 'date_block_num','month','item_vec','cat_vec','shop_vec'
#   input layers--numeric 
    date = Input(shape=(1,), name='date_input')
    month = Input(shape=(1,), name='month_input', dtype='int32')
    
    item = Input(shape=(64,), name='item_input')
    cat = Input(shape=(64,), name='category_input')
    shop = Input(shape=(64,), name='shop_input')

    month_emb = Embedding(input_dim=12, output_dim=3, input_length=1, name='month_emb')(month)
    month_flat = Flatten(name='month_flat')(month_emb)
    
    inputs = Concatenate(axis=-1, name='inputs_concat')([date, month_flat, item, cat, shop])
    inputs_batch = BatchNormalization(name='inputs_batchnorm')(inputs)
    
    preds = Dense(48, activation='relu', name='dense1')(inputs_batch)
#     preds = BatchNormalization(name='batchnorm1')(preds)
#     preds = Dropout(0.1)(preds)
    preds = Dense(16, activation='relu',name='dense2')(preds)
#     preds = Dropout(0.1)(preds)
    preds = Dense(16, activation='relu', name='dense3')(preds)

    # preds = Dense(8,activation ='relu', name='out_nn')(all_inputs)
    preds = Dense(1, activation='relu', name='final_out')(preds)

    return Model(inputs=[date, month, item,cat, shop], outputs=preds)
    

model = build_model()
model.summary()

adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA1)

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred)+0.00001)
   
model.compile(optimizer = adam,loss=rmse, metrics=[rmse])

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

model.fit(inputs, y, batch_size = 2048, epochs=NUM_EPOCHS, callbacks=callbacks, shuffle=True,
          validation_split=0.01)
