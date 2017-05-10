from abc import abstractmethod, ABC
from os.path import join, expanduser

import tensorflow as tf

from .writer import compression


class RecordReader(ABC):
    def __init__(self, tfrecord_pattern, folder=''):
        # Normalize the folder and build the path
        tfrecord_pattern = join(expanduser(folder), tfrecord_pattern)

        # This queue will yield a filename every time it is polled
        file_matcher = tf.train.match_filenames_once(tfrecord_pattern)

        filename_queue = tf.train.string_input_producer(file_matcher)
        reader = tf.TFRecordReader(options=compression)
        tfrecord_key, self._tfrecord_serialized = reader.read(filename_queue)

        self.path = tfrecord_key
        self._read_operation = None

    def read_one(self):
        if self._read_operation is None:
            self._read_operation = self._create_read_operation()
        return self._read_operation

    @abstractmethod
    def _create_read_operation(self):
        """
        Build the specific read operation that should be used to read
        from this TFRecord, one Example at the time
        (is what will be returned by a call to read_one)

        Note: in order to prevent the creation of multiple identical operations,
        this method will be called once, then the operation will be stored
        and returned with very call to read_one
        """
        pass
