"""
First attempt to build a conv encored-decoder architecture
"""
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def build_network(img_in):
    img_encoded = encoder(img_in)
    img_out = decoder(img_encoded)
    return img_out


def encoder(img_in):
    with tf.name_scope('encoder'):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_in)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same', name='img_encoded')(x)
    return encoded


def decoder(encoded):
    with tf.name_scope('decoder'):
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        img_out = Conv2D(3, (3, 3), activation='tanh')(x)
    return img_out


def define_optimizer(img_out, img_true):
    # Trimming necessary due to convolutions (?)
    img_trim = img_true[:, :298, :298]
    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.squared_difference(img_out, img_trim), name="mse")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    return {
        'optimizer': optimizer,
        'cost': cost
    }
