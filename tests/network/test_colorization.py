"""
Train the network on a single sample, a colored checkboard pattern, for 100 epochs
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import color

from koalarization import Colorization, lab_to_rgb, l_to_rgb


NUM_EPOCHS = 10


class TestColorization(unittest.TestCase):
    def test_colorization(self):
        imgs_l, imgs_true_ab, imgs_emb = self._tensors()

        # Build the network and the optimizer step
        col = Colorization(256)
        imgs_ab = col.build(imgs_l, imgs_emb)
        cost = tf.reduce_mean(tf.squared_difference(imgs_ab, imgs_true_ab))
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

        opt_operations = {"cost": cost, "optimizer": optimizer}

        self._run(imgs_l, imgs_ab, imgs_true_ab, opt_operations)

    def _run(self, imgs_l, imgs_ab, imgs_true_ab, opt_operations):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            res = sess.run(
                {"imgs_l": imgs_l, "imgs_ab": imgs_ab, "imgs_true_ab": imgs_true_ab,}
            )

            img_gray = l_to_rgb(res["imgs_l"][0][:, :, 0])
            img_output = lab_to_rgb(res["imgs_l"][0][:, :, 0], res["imgs_ab"][0])
            img_true = lab_to_rgb(res["imgs_l"][0][:, :, 0], res["imgs_true_ab"][0])

            plt.subplot(2, 3, 1)
            plt.imshow(img_gray)
            plt.title("Input (grayscale)")
            plt.axis("off")
            plt.subplot(2, 3, 2)
            plt.imshow(img_output)
            plt.title("Network output")
            plt.axis("off")
            plt.subplot(2, 3, 3)
            plt.imshow(img_true)
            plt.title("Target (original)")
            plt.axis("off")

            for epoch in range(NUM_EPOCHS):
                print("Epoch:", epoch, end=" ")
                res = sess.run(opt_operations)
                print("Cost:", res["cost"])

            res = sess.run(
                {"imgs_l": imgs_l, "imgs_ab": imgs_ab, "imgs_true_ab": imgs_true_ab,}
            )

            img_gray = l_to_rgb(res["imgs_l"][0][:, :, 0])
            img_output = lab_to_rgb(res["imgs_l"][0][:, :, 0], res["imgs_ab"][0])
            img_true = lab_to_rgb(res["imgs_l"][0][:, :, 0], res["imgs_true_ab"][0])

            plt.subplot(2, 3, 4)
            plt.imshow(img_gray)
            plt.title("Input (grayscale)")
            plt.axis("off")
            plt.subplot(2, 3, 5)
            plt.imshow(img_output)
            plt.title("Network output")
            plt.axis("off")
            plt.subplot(2, 3, 6)
            plt.imshow(img_true)
            plt.title("Target (original)")
            plt.axis("off")

            plt.show()

            # Finish off the queue coordinator.
            coord.request_stop()
            coord.join(threads)

    def _tensors(self):
        """
        Create the input and target tensors to feed the network.
        Even if the actual sample is just one, it is batched in a batch of 10
        :return:
        """
        # Image sizes
        width = 128
        height = 64

        # The target image is a simple checkboard pattern
        img = np.zeros((width, height, 3), dtype=np.uint8)
        img[: width // 2, :, 0] = 255
        img[:, height // 2 :, 1] = 255
        img[: width // 2, : height // 2, 2] = 255

        # Simulate a batch of Lab images with size [width, height]
        # and Lab values in the range [-1, 1]
        lab = color.rgb2lab(img).astype(np.float32)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        l = 2 * l / 100 - 1
        l = l.reshape([width, height, 1])
        ab /= 127

        imgs_l, imgs_ab, imgs_emb = tf.train.batch(
            [
                tf.convert_to_tensor(l),
                tf.convert_to_tensor(ab),
                tf.truncated_normal(shape=[1001]),
            ],
            batch_size=10,
        )
        return imgs_l, imgs_ab, imgs_emb


if __name__ == "__main__":
    unittest.main()
