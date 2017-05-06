import multiprocessing
from abc import abstractmethod, ABC
from os.path import join

import tensorflow as tf


class RecordWriter(tf.python_io.TFRecordWriter):
    """
    Wrapper around a TFRecordWriter that adds the methods to serialize an
    image tensor and its properties like the shape.
    """

    def __init__(self, tfrecord_name, dest_folder=''):
        self.path = join(dest_folder, tfrecord_name)
        super().__init__(self.path)

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class RecordReader(ABC):
    def __init__(self, tfrecord_name, folder='.'):
        filename_queue = tf.train.string_input_producer(
            [join(folder, tfrecord_name)])
        reader = tf.TFRecordReader()
        tfrecord_key, self._tfrecord_serialized = reader.read(filename_queue)

        self._read_operation = None

    def read_one(self):
        if self._read_operation is None:
            self._read_operation = self._create_read_operation()
        return self._read_operation

    def read_batch(self, batch_size):
        num_threads = multiprocessing.cpu_count()
        min_after_dequeue = 10 * batch_size
        capacity = min_after_dequeue + (num_threads + 1) * batch_size

        key_batch, image_batch = tf.train.shuffle_batch(
            self.read_one(),
            batch_size,
            capacity,
            min_after_dequeue,
            num_threads,
            allow_smaller_final_batch=True)
        return key_batch, image_batch

    @abstractmethod
    def _create_read_operation(self):
        """
        Build the specific read operation that should be used to read
        from this TFRecord, one Example at the time or in batch
        (is what will be returned by a call to read_one or read_batch)

        Note: in order to prevent the creation of multiple identical operations,
        this method will be called once, then the operation will be stored
        and returned withe very call to read_one and read_batch
        """
        pass
