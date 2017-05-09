import multiprocessing
from abc import ABC

import tensorflow as tf

from .reader import RecordReader


class BatchableRecordReader(RecordReader, ABC):
    """
    For a read operation to be batchable, all of its tensor must have fixed
    sizes at compile time, this rules out e.g. cases where each record
    represents an image and each image can have different size
    """

    def read_batch(self, batch_size):
        # The internet says these values are ok
        num_threads = multiprocessing.cpu_count()
        min_after_dequeue = 10 * batch_size
        capacity = min_after_dequeue + (num_threads + 1) * batch_size

        return tf.train.shuffle_batch(
            self.read_one(),
            batch_size,
            capacity,
            min_after_dequeue,
            num_threads,
            allow_smaller_final_batch=True)
