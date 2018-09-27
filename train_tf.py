import pandas as pd
import numpy as np
import tensorflow as tf
import dnn

# data description and idea
data = pd.read_csv('data/train.csv')

# transform date to datetime object
from datetime import datetime
times = [datetime.strptime(t, '%d.%m.%Y') for t in data.date]
data.date = times

# data aggregation by day_block
X = pd.DataFrame(data.groupby(['date_block_num','shop_id', 'item_id', 'item_price'])['item_cnt_day'].sum()).reset_index()
X['month'] = X.date_block_num % 12

# create train and eval
np.random.seed(654)
msk = np.random.rand(len(X))< 0.9
X_train = X[msk]
X_eval = X[~msk]

# test data preparation
X_test = pd.read_csv('data/test.csv')
X_test['date_block_num'] = 34
X_test['month'] = 11 % 12
X_test.head()

# vocabularies for catgories
shops = pd.read_csv('shops.csv')
items = pd.read_csv('items.csv')
item_cats = pd.read_csv('item_categories.csv')
SHOPS_COUNT = len(shops)
ITEMS_COUNT = len(items)
CATS_COUNT = len(item_cats)

# create feature columns
def create_feature_columns():
    item_col = tf.feature_column.categorical_column_with_identity('item_id', num_buckets = ITEMS_COUNT)
    item_col_emb = tf.feature_column.embedding_column(item_col, CATS_COUNT, initializer=tf.ones_initializer)
    month_col = tf.feature_column.categorical_column_with_identity('month', 12)
    month_col_emb = tf.feature_column.embedding_column(month_col, 2, initializer=tf.ones_initializer)
    shop_col = tf.feature_column.categorical_column_with_identity('shop_id', num_buckets = SHOPS_COUNT)
    shop_col_emb = tf.feature_column.embedding_column(shop_col, 60, initializer=tf.ones_initializer)
    
    return [tf.feature_column.numeric_column('date_block_num'), item_col_emb, month_col_emb, shop_col_emb]

# create input function for train and eval
def make_input_fn(df, num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df['item_cnt_day'],
    batch_size = 64,
    num_epochs = num_epochs,
    shuffle = True
  )

# Feature columns
FEATURES = create_feature_columns()
hidden_units = [48, 16, 16]
LEARNING_RATE = 1e-5
OUTPUT_DIR = './trained_model/'+ 'lr' + str(LEARNING_RATE) + 'units3' + '_'.join([str(s) for s in hidden_units])+'_' + datetime.now().strftime("%dd%H-%M")

# optimizer
gd = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)

# adam = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=1e-4)
ftrl = tf.train.FtrlOptimizer(0.0001)

# # run_configuration 
run_config = tf.estimator.RunConfig(save_checkpoints_steps = 1000)

# exporter for best model
serving_feature_spec = tf.feature_column.make_parse_example_spec(FEATURES)
serving_input_receiver_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec))

exporter = tf.estimator.BestExporter(event_file_pattern='eval/*.tfevents.*',
  name="best_exporter",
  serving_input_receiver_fn=serving_input_receiver_fn,
  exports_to_keep=5)


# DNN linear estimator
# estimator = tf.estimator.DNNLinearCombinedRegressor(model_dir=OUTPUT_DIR, 
#                                                     dnn_optimizer = gd,
#                                                     linear_optimizer = ftrl,
#                                                     dnn_hidden_units = [32, 16, 1],
#                                                    linear_feature_columns=FEATURES[1:2],
# #                                                     batch_norm = True,
#                                                    dnn_feature_columns = [FEATURES[0], FEATURES[3]],
#                                                     loss_reduction = tf.losses.Reduction.MEAN)
# Lienar Regressor                                                   
# estimator = tf.estimator.LinearRegressor(model_dir=OUTPUT_DIR,
#                                         feature_columns=FEATURES,
#                                         loss_reduction = tf.losses.Reduction.MEAN)

# DNN regressor
estimator = dnn.DNNRegressor(model_dir=OUTPUT_DIR,
                                        optimizer = gd,
                                        hidden_units = hidden_units,
                                        batch_norm = True,
                                        feature_columns=FEATURES,
                                        loss_reduction = tf.losses.Reduction.MEAN, config = run_config)

def train_and_eval(NUM_EPOCHS, estimator):
    train_spec = tf.estimator.TrainSpec(input_fn = make_input_fn(X_train, None), max_steps=NUM_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(input_fn = make_input_fn(X_eval, 1), throttle_secs=5, exporters=exporter)
    
    return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


tf.reset_default_graph()
train_and_eval(100000, estimator)