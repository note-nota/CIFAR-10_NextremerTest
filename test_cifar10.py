import tensorflow as tf

from tensorflow.python.keras.datasets import cifar10
from keras.utils import to_categorical

import os
import numpy as np
import matplotlib.pyplot as plt

import model_cifar10

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train,x_test = x_train/255.0,x_test/255.0
y_train,y_test = to_categorical(y_train,10),to_categorical(y_test,10)

model = model_cifar10.create_model()

model.summary()
checkpoint_path = "./checkpoints/ckp"

model.load_weights(checkpoint_path)
loss,acc = model.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
