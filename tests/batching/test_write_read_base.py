"""
Write some simple type examples in a tfrecord, then read them one at the time
or in batch (all the tensors have the same shape, so this is ok)

Run from the top folder as:
python3 -m tests.batching.test_write_read_base
"""
import unittest

import tensorflow as tf

from koalarization.dataset.tfrecords import RecordWriter, BatchableRecordReader


DIR_TFRECORDS = './tests/data/tfrecords'


class BaseTypesRecordWriter(RecordWriter):
    def write_test(self, i):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "string": self._bytes_feature("hey {}".format(i).encode("ascii")),
                    "int": self._int64(42),
                    "float": self._float32(3.14),
                }
            )
        )
        self.write(example.SerializeToString())


class BaseTypesRecordReader(BatchableRecordReader):
    """
    All the tensors returned have fixed shape, so the read operations
    are batchable
    """

    def _create_read_operation(self):
        features = tf.parse_single_example(
            self._tfrecord_serialized,
            features={
                "string": tf.FixedLenFeature([], tf.string),
                "int": tf.FixedLenFeature([], tf.int64),
                "float": tf.FixedLenFeature([], tf.float32),
            },
        )

        return {
            "string": features["string"],
            "int": features["int"],
            "float": features["float"],
        }


class TestBaseRecords(unittest.TestCase):
    number_of_records = 5
    samples_per_record = 10

    def test_base_records(self):
        # WRITING
        for i in range(self.number_of_records):
            record_name = "base_type_{}.tfrecord".format(i)
            with BaseTypesRecordWriter(record_name, DIR_TFRECORDS) as writer:
                for j in range(self.samples_per_record):
                    writer.write_test(i * self.number_of_records + j)

        # READING
        # Important: read_batch MUST be called before start_queue_runners,
        # otherwise the internal shuffle queue gets created but its
        # threads won't start

        reader = BaseTypesRecordReader("base_type_*.tfrecord", DIR_TFRECORDS)
        read_one_example = reader.read_operation
        read_batched_examples = reader.read_batch(50)

        with tf.Session() as sess:
            sess.run(
                [tf.global_variables_initializer(), tf.local_variables_initializer()]
            )

            # Coordinate the queue of tfrecord files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Reading examples sequentially one by one
            for j in range(50):
                fetches = sess.run(read_one_example)
                print("Read:", fetches)

            # Reading a batch of examples
            results = sess.run(read_batched_examples)
            for i in range(len(results["string"])):
                print(
                    results["string"][i],
                    results["int"][i],
                    results["float"][i],
                    sep="\t",
                )

            # Finish off the queue coordinator.
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    unittest.main()
