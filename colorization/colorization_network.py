"""
Given the L channel of an Lab image (range [-1, +1]), output a prediction over
the a and b channels in the range [-1, 1].
In the neck of the conv-deconv network use the features from a feature extractor
(e.g. Inception) and fuse them with the conv output.

When using
l, emb, ab = sess.run([image_l, image_embedding, image_ab])

The function l_to_rgb converts the numpy array l into an rgb image.
The function lab_to_rgb converts the numpy arrays l and b into an rgb image.
"""
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from skimage import color

from colorization.fusion_layer import fusion


def colorization(img_l, img_emb):
    img_encoded = encoder(img_l)
    img_fused = fusion(img_encoded, img_emb)
    img_ab = decoder(img_fused)
    return img_ab


def encoder(img_l):
    with tf.name_scope('encoder'):
        x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(
            img_l)
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
        x = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
        img_ab = UpSampling2D((2, 2))(x)
    return img_ab


def define_optimizer(img_ab_out, img_ab_true):
    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.squared_difference(img_ab_out, img_ab_true),
                          name="mse")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # Metrics for tensorboard
    with tf.name_scope('summaries'):
        tf.summary.scalar('cost', cost)

    return {
        'optimizer': optimizer,
        'cost': cost
    }


def l_to_rgb(img_l):
    lab = np.squeeze(255 * (img_l + 1) / 2)
    return color.gray2rgb(lab) / 255


def lab_to_rgb(img_l, img_ab):
    lab = np.empty([*img_l.shape[0:2], 3])
    lab[:, :, 0] = np.squeeze(((img_l + 1) * 50))
    lab[:, :, 1:] = img_ab * 127
    return color.lab2rgb(lab)
