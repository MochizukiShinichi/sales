# predict and write result
from __future__ import absolute_import
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
from keras.backend import sqrt

# vocabularies
X_test = pd.read_csv('X_test.csv')

x_test = X_test.values
inputs_test = [x_test[:,i].tolist() for i in range(x_test.shape[1])]

from keras.losses import mean_squared_error

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

model = load_model('trained_model/lr0.001_09d10-45/weights-improvement-01-0.963553.hdf5', {'rmse':rmse, 'sqrt':sqrt})

y_out = model.predict(inputs_test, verbose=1).flatten().tolist()
y_out = [20 if i>20 else i for i in y_out]

import csv
with open('predictions.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(('ID','item_cnt_month'))
    for i in range(len(y_out)):
        writer.writerow((i, y_out[i]))