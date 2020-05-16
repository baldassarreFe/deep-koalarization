"""
Read and show single images from a tfrecord

Run from the top folder as:
python3 -m tests.batching.test_write_read_single_image
"""
import time
import unittest
from os.path import basename

import matplotlib.pyplot as plt
import tensorflow as tf

from koalarization.dataset.tfrecords import (
    SingleImageRecordWriter,
    SingleImageRecordReader,
)
from koalarization.dataset.tfrecords import queue_single_images_from_folder


DIR_RESIZED = './tests/data/resized'
DIR_TFRECORDS = './tests/data/tfrecords'


class TestSingleImageWriteRead(unittest.TestCase):
    def test_single_image_write_read(self):
        self._single_image_write()
        self._single_image_read()

    def _single_image_write(self):
        # Create the queue operations
        img_key, img_tensor, _ = queue_single_images_from_folder(DIR_RESIZED)

        # Create a writer to write_image the images
        single_writer = SingleImageRecordWriter("single_images.tfrecord", DIR_TFRECORDS)

        # Start a new session to run the operations
        with tf.Session() as sess:
            sess.run(
                [tf.global_variables_initializer(), tf.local_variables_initializer()]
            )

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            count = 0
            start_time = time.time()
            try:
                while not coord.should_stop():
                    key, img = sess.run([img_key, img_tensor])
                    single_writer.write_image(key, img)
                    print("Written: {}".format(key))
                    count += 1
            except tf.errors.OutOfRangeError:
                # The string_input_producer queue ran out of strings
                pass
            finally:
                # Ask the threads (filename queue) to stop.
                coord.request_stop()
                print(
                    "Finished writing {} images in {:.2f}s".format(
                        count, time.time() - start_time
                    )
                )

            # Wait for threads to finish.
            coord.join(threads)

        single_writer.close()

    def _single_image_read(self):
        # Important: read_batch MUST be called before start_queue_runners,
        # otherwise the internal shuffle queue gets created but its
        # threads won't start
        irr = SingleImageRecordReader("single_images.tfrecord", DIR_TFRECORDS)
        read_one_example = irr.read_operation
        read_batched_examples = irr.read_batch(10)

        with tf.Session() as sess:
            sess.run(
                [tf.global_variables_initializer(), tf.local_variables_initializer()]
            )

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Reading images sequentially one by one
            for i in range(6):
                res = sess.run(read_one_example)
                plt.subplot(2, 3, i + 1)
                plt.imshow(res["image"])
                plt.axis("off")
                print("Read", basename(res["key"]))
            plt.show()

            # Reading images in batch
            res = sess.run(read_batched_examples)
            print(res["key"], res["image"].shape)

            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    unittest.main()
