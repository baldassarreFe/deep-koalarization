"""
Write some fixed size examples in a tfrecord, then read them one at the time
or in batch (all the tensors have the same shape, so this is ok)

Run from the top folder as:
python3 -m tests.batching.test_write_read_fixed
"""
import unittest

import numpy as np
import tensorflow as tf

from koalarization.dataset.tfrecords import RecordWriter, BatchableRecordReader


DIR_TFRECORDS = './tests/data/tfrecords'


class FixedSizeTypesRecordWriter(RecordWriter):
    def write_test(self):
        # Fixed size lists
        list_ints = np.array([4, 8, 15, 16, 23, 42], dtype=np.int64)
        list_floats = np.array([2.71, 3.14], dtype=np.float32)

        # Fixed size matrices (will be flattened before serializing)
        mat_ints = np.arange(6, dtype=np.int64).reshape(2, 3)
        mat_floats = np.random.random((3, 2)).astype(np.float32)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "list_ints": self._int64_list(list_ints),
                    "list_floats": self._float32_list(list_floats),
                    "mat_ints": self._int64_list(mat_ints.flatten()),
                    "mat_floats": self._float32_list(mat_floats.flatten()),
                }
            )
        )
        self.write(example.SerializeToString())


class FixedSizeTypesRecordReader(BatchableRecordReader):
    """
    All the tensors returned have fixed shape, so the read operations
    are batchable
    """

    def _create_read_operation(self):
        features = tf.parse_single_example(
            self._tfrecord_serialized,
            features={
                "list_ints": tf.FixedLenFeature([6], tf.int64),
                "list_floats": tf.FixedLenFeature([2], tf.float32),
                "mat_ints": tf.FixedLenFeature([6], tf.int64),
                "mat_floats": tf.FixedLenFeature([6], tf.float32),
            },
        )

        return {
            "list_ints": features["list_ints"],
            "list_floats": features["list_floats"],
            "mat_ints": tf.reshape(features["mat_ints"], [2, 3]),
            "mat_floats": tf.reshape(features["mat_floats"], [3, 2]),
        }


class TestFixedSizeRecords(unittest.TestCase):
    def test_fixed_size_record(self):
        # WRITING
        with FixedSizeTypesRecordWriter("fixed_size.tfrecord", DIR_TFRECORDS) as writer:
            writer.write_test()
            writer.write_test()

        # READING
        # Important: read_batch MUST be called before start_queue_runners,
        # otherwise the internal shuffle queue gets created but its
        # threads won't start
        reader = FixedSizeTypesRecordReader("fixed_size.tfrecord", DIR_TFRECORDS)
        read_one_example = reader.read_operation
        read_batched_examples = reader.read_batch(4)

        with tf.Session() as sess:
            sess.run(
                [tf.global_variables_initializer(), tf.local_variables_initializer()]
            )

            # Coordinate the queue of tfrecord files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Reading examples sequentially one by one
            for j in range(3):
                fetches = sess.run(read_one_example)
                print("Read:", fetches)

            # Reading a batch of examples
            results = sess.run(read_batched_examples)
            print(results)

            # Finish off the queue coordinator.
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    unittest.main()
