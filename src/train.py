import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_mnist(as_cnn=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if as_cnn:
        x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
        x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    else:
        x_train = x_train.reshape((-1, 784)).astype('float32') / 255.0
        x_test = x_test.reshape((-1, 784)).astype('float32') / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test