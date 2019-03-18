import tensorflow as tf
import pandas
import numpy as np
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import tables

NAME = "network_pid".format(int(time()))
X = pandas.read_hdf("particle_data_big_kaon0_pion1_proton2.h5", start=0, stop=1000000)
X = X[X.TrackP <= 7000]
y = X.pi_TRUEID
X= X.drop("pi_TRUEID", axis = 1)
# X= X.drop("pi_TRACK_time", axis = 1)
# X= X.drop("pi_TRACK_time_err", axis = 1)


print(X)
print(y)
y = np.array(y, ndmin=2).T
X = np.array(X, ndmin=2)
for i in range(len(y)):
    if y[i] == (2):
        y[i] = 0
np.save("y_pion1_rest0", y)
np.save("X_pion1_rest0", X)