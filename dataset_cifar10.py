from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

def get_data():
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()

    x_train,x_test = x_train/255.0,x_test/255.0
    y_train,y_test = to_categorical(y_train,10),to_categorical(y_test,10)

    return (x_train, np.where(y_train < 0.5, 0.02, 0.82)),(x_test,y_test)
