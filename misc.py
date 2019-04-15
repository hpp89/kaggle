from keras import layers
from keras.models import Model, Input
from tensorflow.contrib.layers import xavier_initializer
from keras.regularizers import l2


def Conv2D(inputs, filters, name, kernel_size=3, activation=None, weight_decay=.0):
    output = layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',
        name=name, activation=activation, kernel_initializer=xavier_initializer(), kernel_regularizer=l2(weight_decay))(
        inputs)

    return output


def Conv2D_with_strides(inputs, filters, name, strides, kernel_size=3, activation=None, weight_decay=.0):
    output = layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(strides, strides),
        padding='same', name=name, activation=activation, kernel_initializer=xavier_initializer(),
        kernel_regularizer=l2(weight_decay))(inputs)

    return output


def MaxPooling2D(inputs, name, pool_size=2, strides=2):
    output = layers.MaxPooling2D(pool_size=(pool_size, pool_size), strides=(strides, strides), padding='same',
        name=name)(inputs)

    return output


def GlobalAveragePooling2D(inputs):
    output = layers.GlobalAveragePooling2D()(inputs)

    return output


def Dense(inputs, units, name=None, activation=None, weight_decay=.0):
    output = layers.Dense(units=units, activation=activation, name=name, kernel_initializer=xavier_initializer(),
        kernel_regularizer=l2(weight_decay))(inputs)

    return output
