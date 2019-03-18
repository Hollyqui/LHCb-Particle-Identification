import tensorflow as tf
import pandas
import numpy as np
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import tables
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


NAME = "network_pid".format(int(time()))

y = np.load("y_pion1_rest0.npy")
X = np.load("X_pion1_rest0.npy")

test_X = X
X = tf.keras.utils.normalize(
        X,
        axis=-1,
        order=2
        )
class_weight = {0:9,
                1:6
                }

print(X)
print(y)

model = keras.Sequential([
    keras.layers.Dense(input_shape=(31,), units=2),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.sigmoid)
])

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# sets optimization function, learning rate and loss function for network
model.compile(optimizer= tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# trains the network for 1000 epochs with 500 iterations per epoch
model.fit(X,
          y,
          epochs = 100,
          callbacks=[tensorboard],
          batch_size=100,
          shuffle=True,
          validation_split=0.1,
          # class_weight = class_weight
          )


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)




predictions = model.predict(X)
# print(predictions)
k= 0
confusion_matrix = []
for i in range(2):
    confusion_matrix.append([])
    for j in range(2):
        confusion_matrix[i].append(0)

print(predictions)
print(y)
roc_predictions = predictions.T[1]
for i in predictions:
   prediction = np.argmax(predictions[k])
   real_result = y[k][0]
   confusion_matrix[prediction][real_result]= confusion_matrix[prediction][real_result]+1
   k = k + 1
#confusion_matrix[0][9] = 'test'
confusion_matrix = np.array(confusion_matrix)
print(confusion_matrix)
print(real_result)

np.save("predictions", predictions)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y,roc_predictions)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()