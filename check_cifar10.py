import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

import dataset_cifar10

(x_train,y_train),(x_test,y_test) = dataset_cifar10.get_data()

print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

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
