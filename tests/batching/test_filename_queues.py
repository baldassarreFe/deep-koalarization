import time
import unittest

import tensorflow as tf

from koalarization.dataset.tfrecords import queue_single_images_from_folder


DIR_RESIZED = './tests/data/resized/'


class TestFilenameQueues(unittest.TestCase):
    def test_one(self):
        """Load all images from a folder once and print the result."""
        # Create the queue operations
        image_key, image_tensor, image_shape = queue_single_images_from_folder(
            DIR_RESIZED
        )

        # Start a new session to run the operations
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            count = 0
            start_time = time.time()

            # These are the only lines where something happens:
            # we execute the operations to get one image and print everything.
            try:
                while not coord.should_stop():
                    key, img, shape = sess.run([image_key, image_tensor, image_shape])
                    print(key)
                    count += 1
            except tf.errors.OutOfRangeError:
                # It's all right, it's just the string_input_producer queue telling
                # us that it has run out of strings
                pass
            finally:
                # Ask the threads (filename queue) to stop.
                coord.request_stop()
                print(
                    "Finished listing {} pairs in {:.2f}s".format(
                        count, time.time() - start_time
                    )
                )

            # Wait for threads to finish.
            coord.join(threads)


if __name__ == "__main__":
    unittest.main()
