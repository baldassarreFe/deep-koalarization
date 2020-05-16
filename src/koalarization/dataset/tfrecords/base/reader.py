from abc import abstractmethod, ABC
from os.path import join, expanduser

import tensorflow as tf

from .writer import compression


class RecordReader(ABC):
    """
    A class to read examples from all the TFRecord matching a certain
    filename pattern. The implementation of the read operation is left
    to the subclasses, while the logic to queue all the record files as
    a single data source is provided here.
    """

    def __init__(self, tfrecord_pattern, folder=""):
        # Normalize the folder and build the path
        tfrecord_pattern = join(expanduser(folder), tfrecord_pattern)

        # This queue will yield a filename every time it is polled
        file_matcher = tf.train.match_filenames_once(tfrecord_pattern)

        filename_queue = tf.train.string_input_producer(file_matcher)
        reader = tf.TFRecordReader(options=compression)
        tfrecord_key, self._tfrecord_serialized = reader.read(filename_queue)

        self._path = tfrecord_key
        self._read_operation = None

    @property
    def read_operation(self):
        if self._read_operation is None:
            self._read_operation = self._create_read_operation()
        return self._read_operation

    @abstractmethod
    def _create_read_operation(self):
        """
        Build the specific read operation that should be used to read
        from the TFRecords in the queue, one Example at the time

        Note: in order to prevent the creation of multiple identical operations,
        this method will be called once, then the operation will be available
        as ``read_operation``
        """
        pass
