"""
Read and show single images from a tfrecord

Run from the top folder as:
python3 -m tests.batching.test_write_read_lab_image
"""
import time
import unittest
from os.path import basename

import matplotlib.pyplot as plt
import tensorflow as tf

from koalarization import l_to_rgb
from koalarization import lab_to_rgb
from koalarization.dataset.tfrecords import LabImageRecordReader
from koalarization.dataset.tfrecords import LabImageRecordWriter
from koalarization.dataset.tfrecords import queue_single_images_from_folder


DIR_RESIZED = './tests/data/resized/'
DIR_TFRECORDS = './tests/data/tfrecords'


class TestLabImageWriteRead(unittest.TestCase):
    def test_lab_image_write_read(self):
        self._lab_image_write()
        self._lab_image_read()

    def _lab_image_write(self):
        # Create the queue operations
        img_key, img_tensor, _ = queue_single_images_from_folder(DIR_RESIZED)
        img_emb = tf.truncated_normal(shape=[1001])

        # Create a writer to write_image the images
        lab_writer = LabImageRecordWriter("test_lab_images.tfrecord", DIR_TFRECORDS)

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
                    key, img, emb = sess.run([img_key, img_tensor, img_emb])
                    lab_writer.write_image(key, img, emb)
                    print("Written: {}".format(key))
                    count += 1
                    # Just write 10 images
                    if count > 10:
                        break
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

        lab_writer.close()

    def _lab_image_read(self):
        # Important: read_batch MUST be called before start_queue_runners,
        # otherwise the internal shuffle queue gets created but its
        # threads won't start
        irr = LabImageRecordReader("test_lab_images.tfrecord", DIR_TFRECORDS)
        read_one_example = irr.read_operation
        read_batched_examples = irr.read_batch(20)

        with tf.Session() as sess:
            sess.run(
                [tf.global_variables_initializer(), tf.local_variables_initializer()]
            )

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Reading images sequentially one by one
            for i in range(0, 12, 2):
                res = sess.run(read_one_example)
                img = lab_to_rgb(res["image_l"], res["image_ab"])
                img_gray = l_to_rgb(res["image_l"])
                plt.subplot(3, 4, i + 1)
                plt.imshow(img_gray)
                plt.axis("off")
                plt.subplot(3, 4, i + 2)
                plt.imshow(img)
                plt.axis("off")
                print("Read", basename(res["image_name"]))
            plt.show()

            # Reading images in batch
            res = sess.run(read_batched_examples)
            print(
                res["image_name"],
                res["image_l"].shape,
                res["image_ab"].shape,
                res["image_embedding"].shape,
                sep="\n",
            )

            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    unittest.main()
