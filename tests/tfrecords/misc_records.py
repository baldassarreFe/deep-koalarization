"""
A series of tfrecord writers and readers that deal with different data types

See the usages in:
- write_read_base.py
- write_read_fixed.py
- write_read_variable.py
"""
import numpy as np
import tensorflow as tf

from dataset.tfrecords import RecordReader, BatchableRecordReader, RecordWriter


class BaseTypesRecordWriter(RecordWriter):
    def write_test(self):
        example = tf.train.Example(features=tf.train.Features(feature={
            'string': self._bytes_feature(b'hey'),
            'int': self._int64(42),
            'float': self._float32(3.14),
        }))
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
                'string': tf.FixedLenFeature([], tf.string),
                'int': tf.FixedLenFeature([], tf.int64),
                'float': tf.FixedLenFeature([], tf.float32),
            })

        return {
            'string': features['string'],
            'int': features['int'],
            'float': features['float'],
        }


class FixedSizeTypesRecordWriter(RecordWriter):
    def write_test(self):
        # Fixed size lists
        list_ints = np.array([4, 8, 15, 16, 23, 42], dtype=np.int64)
        list_floats = np.array([2.71, 3.14], dtype=np.float32)

        # Fixed size matrices (will be flattened before serializing)
        mat_ints = np.arange(6, dtype=np.int64).reshape(2, 3)
        mat_floats = np.random.random((3, 2)).astype(np.float32)

        example = tf.train.Example(features=tf.train.Features(feature={
            'list_ints': self._int64_list(list_ints),
            'list_floats': self._float32_list(list_floats),
            'mat_ints': self._int64_list(mat_ints.flatten()),
            'mat_floats': self._float32_list(mat_floats.flatten()),
        }))
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
                'list_ints': tf.FixedLenFeature([6], tf.int64),
                'list_floats': tf.FixedLenFeature([2], tf.float32),
                'mat_ints': tf.FixedLenFeature([6], tf.int64),
                'mat_floats': tf.FixedLenFeature([6], tf.float32),
            })

        return {
            'list_ints': features['list_ints'],
            'list_floats': features['list_floats'],
            'mat_ints': tf.reshape(features['mat_ints'], [2, 3]),
            'mat_floats': tf.reshape(features['mat_floats'], [3, 2]),
        }


class VariableSizeTypesRecordWriter(RecordWriter):
    """
    The tensors returned don't have the same shape, so the read operations
    are not batchable. Also here we have types that are not int64 or
    float32, so we serialize them as raw bytes.
    """

    def write_test(self):
        # Shape must be an int32 for the reshape operation during reading
        # to succeed (we'll need to serialize the shape too)
        shape = np.random.random_integers(2, 4, 2).astype(np.int32)

        # Variable size matrices of uint8 (like an image) and float16
        mat_ints = np.random.random_integers(0, 255, shape).astype(np.uint8)
        mat_floats = np.random.random(shape).astype(np.float16)

        example = tf.train.Example(features=tf.train.Features(feature={
            'shape': self._bytes_feature(shape.tobytes()),
            'mat_ints': self._bytes_feature(mat_ints.tobytes()),
            'mat_floats': self._bytes_feature(mat_floats.tobytes())
        }))
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
                'shape': tf.FixedLenFeature([], tf.string),
                'mat_ints': tf.FixedLenFeature([], tf.string),
                'mat_floats': tf.FixedLenFeature([], tf.string)
            })

        shape = tf.decode_raw(features['shape'], tf.int32)

        mat_ints = tf.decode_raw(features['mat_ints'], tf.uint8)
        mat_ints = tf.reshape(mat_ints, shape)

        mat_floats = tf.decode_raw(features['mat_floats'], tf.float16)
        mat_floats = tf.reshape(mat_floats, shape)

        return {
            'shape': shape,
            'mat_ints': mat_ints,
            'mat_floats': mat_floats
        }
