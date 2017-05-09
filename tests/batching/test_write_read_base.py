"""
Write some simple type examples in a tfrecord, then read them one at the time
or in batch (all the tensors have the same shape, so this is ok)

Run from the top folder as:
python3 -m tests.batching.test_write_read_base
"""
import unittest

import tensorflow as tf

from dataset.shared import dir_tfrecord
from .misc_records import BaseTypesRecordReader, BaseTypesRecordWriter


class TestBaseRecords(unittest.TestCase):
    def test_base_records(self):
        # WRITING
        with BaseTypesRecordWriter('base_type.tfrecord',
                                   dir_tfrecord) as writer:
            for i in range(2):
                writer.write_test()

        # READING
        # Important: read_batch MUST be called before start_queue_runners,
        # otherwise the internal shuffle queue gets created but its
        # threads won't start

        reader = BaseTypesRecordReader('base_type.tfrecord', dir_tfrecord)
        read_batched_examples = reader.read_batch(4)
        read_one_example = reader.read_one()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Coordinate the queue of tfrecord files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Reading examples sequentially one by one
            for j in range(3):
                fetches = sess.run(read_one_example)
                print('Read:', fetches)

            # Reading a batch of examples
            results = sess.run(read_batched_examples)
            print(results)

            # Finish off the queue coordinator.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    unittest.main()
