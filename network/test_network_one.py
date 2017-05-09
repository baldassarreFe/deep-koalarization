import glob
import os
from os.path import expanduser

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from matplotlib import pyplot as plt

from network.network_one import build_network
from network.network_one import define_optimizer

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

if __name__ == '__main__':

    sess = tf.Session()
    K.set_session(sess)

    directory_output = expanduser('~/imagenet/resized_')
    directory_input = expanduser('~/imagenet/filtered_')

    images_in = []
    images_out = []

    for filename_image_out in os.listdir(directory_output):
        if filename_image_out.endswith("jpeg"):
            path_to_images_in = glob.glob(
                directory_input + "/" + filename_image_out.split(".")[0] + "*")
            path_to_image_out = os.path.join(directory_output,
                                             filename_image_out)
            image_out = Image.open(path_to_image_out)
            image_out = np.array(image_out)
            for path_to_image_in in path_to_images_in:
                image_in = Image.open(path_to_image_in)
                image_in = np.array(image_in)
                images_in.append(image_in)
                images_out.append(image_out)

    images_in = np.array(images_in)
    images_out = np.array(images_out)

    images_out_processed = (images_out - 127.5) / 127.5
    images_in_processed = (images_in - 127.5) / 127.5

    img_in = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    img_true = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

    img_out = build_network(img_in)
    operations = define_optimizer(img_out, img_true)

    training_epochs = 2
    batch_size = 8
    total_batches = len(images_in) // batch_size

    idx = np.arange(len(images_in))

    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        for epoch in range(training_epochs):
            np.random.shuffle(idx)
            images_in_processed = images_in_processed[idx]
            images_out_processed = images_out_processed[idx]
            print("\nepoch:", epoch)
            for i in range(total_batches):
                batch_in = images_in_processed[
                           i * batch_size:(i + 1) * batch_size]
                batch_out = images_out_processed[
                            i * batch_size:(i + 1) * batch_size]
                res = sess.run(operations,
                               feed_dict={img_in: batch_in,
                                          img_true: batch_out})
                print(str(i) + ", Cost =  ", res['cost'])

    # Evaluation
    image_in = images_in_processed
    image_out = images_out_processed
    with sess.as_default():
        pred, true = sess.run([img_out, img_true],
                              feed_dict={img_in: image_in, img_true: image_out})

    for k in range(10):
        original = ((127.5 * image_out[k]) + 127.5).astype("uint8")
        estimation = ((127.5 * pred[k]) + 127.5).astype("uint8")
        filtered = ((127.5 * image_in[k]) + 127.5).astype("uint8")
        # Four axes, returned as a 2-d array
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(original)
        axarr[0].set_title('Original')
        axarr[1].imshow(filtered)
        axarr[1].set_title('Filtered')
        axarr[2].imshow(estimation)
        axarr[2].set_title('Reconstructed')

        plt.show()
