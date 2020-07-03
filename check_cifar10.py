import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from keras.utils import to_categorical

import os
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
y_train,y_test = to_categorical(y_train,10),to_categorical(y_test,10)

print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)

LABELS = ('airplane', 'mobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse','ship', 'truck')

def index_to_label(idx):
  if idx < len(LABELS):
    return LABELS[idx]
  else:
    return None

def vector_to_label(v):
  idx = np.argmax(v)
  return index_to_label(idx)

plt.clf()
for i in range(0, 40):
  plt.subplot(5, 8, i+1)
  pixels = x_train[i,:,:,:]
  plt.title(vector_to_label(y_train[i]), fontsize=8)
  fig = plt.imshow(pixels)
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)

plt.savefig('cifar10_image_train.png')
