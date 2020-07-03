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

history = model.fit(x_train, y_train, validation_split=0.25, epochs=5, batch_size=16, verbose=1)

model.save_weights('./checkpoints/my_checkpoint')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
