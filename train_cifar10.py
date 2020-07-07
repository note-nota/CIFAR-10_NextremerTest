import sys
import argparse
import numpy as np
import tensorflow as tf
import mlflow.keras
import matplotlib.pyplot as plt

import model_cifar10
import dataset_cifar10

mlflow.keras.autolog()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--epochs', default=5, type=int, help='epoch size')

def main(argv):
    args = parser.parse_args(argv[1:])
    (x_train,y_train),(x_test,y_test) = dataset_cifar10.get_data()

    model = model_cifar10.create_model()

    history = model.fit(x_train, y_train, validation_split=0.25, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    model.save_weights('./checkpoints/ckp')

    # For Checking
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(args.epochs)

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

if __name__ == '__main__':
    main(sys.argv)
