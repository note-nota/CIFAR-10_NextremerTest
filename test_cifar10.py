import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

import model_cifar10
import dataset_cifar10

(x_train,y_train),(x_test,y_test) = dataset_cifar10.get_data()

model = model_cifar10.create_model()

model.summary()
checkpoint_path = "./checkpoints/ckp"

model.load_weights(checkpoint_path)
loss,acc = model.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
