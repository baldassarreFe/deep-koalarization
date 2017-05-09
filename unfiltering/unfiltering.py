"""
Given an RGB image (range [-1, +1]), output an unfiltered version of the image
in the range [-1, 1]. In the neck of the conv-deconv network use the features
from a feature extractor (e.g. Inception)
"""
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D

from colorization.fusion_layer import fusion


def unfiltering(img_in, img_emb):
    img_encoded = encoder(img_in)
    img_fused = fusion(img_encoded, img_emb)
    img_out = decoder(img_fused)
    return img_out


def encoder(img_in):
    with tf.name_scope('encoder'):
        x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(
            img_in)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    return x


def decoder(encoded):
    with tf.name_scope('decoder'):
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
        img_out = UpSampling2D((2, 2))(x)
    return img_out


def define_optimizer(img_out, img_true):
    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.squared_difference(img_out, img_true), name="mse")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # Metrics for tensorboard
    with tf.name_scope('summaries'):
        tf.summary.scalar('cost', cost)

    return {
        'optimizer': optimizer,
        'cost': cost
    }
