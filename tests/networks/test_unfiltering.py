import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from unfiltering.unfiltering import unfiltering, define_optimizer


class TestUnfiltering(unittest.TestCase):
    def test_absolute(self):
        imgs_emb, imgs_in, imgs_true = self._common_tensors()

        # Build the network and the optimizer step
        imgs_out = unfiltering(imgs_in, imgs_emb)
        opt_operations = define_optimizer(imgs_out, imgs_true)

        self._run(imgs_in, imgs_out, imgs_true, opt_operations)

    def test_relative(self):
        imgs_emb, imgs_in, imgs_true = self._common_tensors()
        imgs_true += imgs_in

        # Build the network and the optimizer step
        imgs_delta = unfiltering(imgs_in, imgs_emb)
        imgs_out = imgs_in - imgs_delta
        opt_operations = define_optimizer(imgs_out, imgs_true)

        self._run(imgs_in, imgs_out, imgs_true, opt_operations)

    def _run(self, imgs_in, imgs_out, imgs_true, opt_operations):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            res = sess.run({
                'imgs_in': imgs_in,
                'imgs_out': imgs_out,
                'imgs_true': imgs_true,
            })

            plt.subplot(2, 3, 1)
            plt.imshow((res['imgs_in'][0] + 1) / 2)
            plt.title('Input (filtered)')
            plt.axis('off')
            plt.subplot(2, 3, 2)
            plt.imshow((res['imgs_out'][0] + 1) / 2)
            plt.title('Output (our unfiltered)')
            plt.axis('off')
            plt.subplot(2, 3, 3)
            plt.imshow((res['imgs_true'][0] + 1) / 2)
            plt.title('Target (unfiltered)')
            plt.axis('off')

            for epoch in range(100):
                print('Epoch:', epoch, end=' ')
                res = sess.run(opt_operations)
                print('Cost:', res['cost'])

            res = sess.run({
                'imgs_in': imgs_in,
                'imgs_out': imgs_out,
                'imgs_true': imgs_true,
            })

            plt.subplot(2, 3, 4)
            plt.imshow((res['imgs_in'][0] + 1) / 2)
            plt.title('Input (filtered)')
            plt.axis('off')
            plt.subplot(2, 3, 5)
            plt.imshow((res['imgs_out'][0] + 1) / 2)
            plt.title('Output (our unfiltered)')
            plt.axis('off')
            plt.subplot(2, 3, 6)
            plt.imshow((res['imgs_true'][0] + 1) / 2)
            plt.title('Target (unfiltered)')
            plt.axis('off')

            plt.show()

            # Finish off the queue coordinator.
            coord.request_stop()
            coord.join(threads)

    def _common_tensors(self):
        # Image sizes
        width = 128
        height = 64
        # The target image is a simple checkboard pattern
        img_true = np.ones((width, height, 3), dtype=np.float32)
        img_true[:width // 2, :, 0] = -1
        img_true[:, height // 2:, 1] = -1
        img_true[:width // 2, :height // 2, 2] = -1
        # Simulate a batch of RGB images with size [width, height, 3]
        # and pixel values in the range [-1, 1]
        imgs_in, imgs_true, imgs_emb = tf.train.batch([
            tf.truncated_normal(shape=[width, height, 3], stddev=0.5),
            tf.convert_to_tensor(img_true),
            tf.truncated_normal(shape=[1001])],
            batch_size=3
        )
        return imgs_emb, imgs_in, imgs_true


if __name__ == '__main__':
    unittest.main()
