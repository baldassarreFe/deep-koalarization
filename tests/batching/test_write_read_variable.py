"""
Write some variable size examples in a tfrecord, then read them one at the time
(all the tensors have differente shapes, so we can't batch them)

Run from the top folder as:
python3 -m tests.batching.test_write_read_variable
"""
import unittest

import numpy as np
import tensorflow as tf

from koalarization.dataset.tfrecords import RecordWriter, RecordReader


DIR_TFRECORDS = './tests/data/tfrecords'


class VariableSizeTypesRecordWriter(RecordWriter):
    """
    The tensors returned don't have the same shape, so the read operations
    are not batchable. Also here we have types that are not int64 or
    float32, so we serialize them as raw bytes.
    """

    def write_test(self):
        # Shape must be an int32 for the reshape operation during reading
        # to succeed (we'll need to serialize the shape too)
        shape = np.random.randint(2, 4, 2, dtype=np.int32)

        # Variable size matrices of uint8 (like an image) and float16
        mat_ints = np.random.randint(0, 255, shape, dtype=np.uint8)
        mat_floats = np.random.random(shape).astype(np.float16)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "shape": self._bytes_feature(shape.tobytes()),
                    "mat_ints": self._bytes_feature(mat_ints.tobytes()),
                    "mat_floats": self._bytes_feature(mat_floats.tobytes()),
                }
            )
        )
        self.write(example.SerializeToString())


class VariableSizeTypesRecordReader(RecordReader):
    """
    All the tensors returned have fixed shape, so the read operations
    are batchable
    """

    def _create_read_operation(self):
        features = tf.parse_single_example(
            self._tfrecord_serialized,
            features={
                "shape": tf.FixedLenFeature([], tf.string),
                "mat_ints": tf.FixedLenFeature([], tf.string),
                "mat_floats": tf.FixedLenFeature([], tf.string),
            },
        )

        shape = tf.decode_raw(features["shape"], tf.int32)

        mat_ints = tf.decode_raw(features["mat_ints"], tf.uint8)
        mat_ints = tf.reshape(mat_ints, shape)

        mat_floats = tf.decode_raw(features["mat_floats"], tf.float16)
        mat_floats = tf.reshape(mat_floats, shape)

        return {"shape": shape, "mat_ints": mat_ints, "mat_floats": mat_floats}


class TestVariableSizeRecords(unittest.TestCase):
    def test_variable_size_record(self):
        # WRITING
        with VariableSizeTypesRecordWriter("variable.tfrecord", DIR_TFRECORDS) as writer:
            for i in range(2):
                writer.write_test()

        # READING
        reader = VariableSizeTypesRecordReader("variable.tfrecord", DIR_TFRECORDS)
        read_one_example = reader.read_operation

        with tf.Session() as sess:
            sess.run(
                [tf.global_variables_initializer(), tf.initialize_local_variables()]
            )

            # Coordinate the queue of tfrecord files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Reading examples sequentially one by one
            for j in range(3):
                fetches = sess.run(read_one_example)
                print("Read:", fetches)

            # Finish off the queue coordinator.
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    unittest.main()
