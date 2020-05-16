import multiprocessing

import tensorflow as tf

from .reader import RecordReader


class BatchableRecordReader(RecordReader):
    """
    Provides the same functionality as the parent RecordReader, adding the
    possibility to get a batched version of the ``read_operation``

    For a read operation to be batchable, all of its tensor must have fixed
    sizes at compile time, this rules out e.g. cases where each record
    represents an image and each image can have different size
    """

    def read_batch(self, batch_size, shuffle=False):
        # Recommended configuration for these parameters (found online)
        num_threads = multiprocessing.cpu_count()
        min_after_dequeue = 10 * batch_size
        capacity = min_after_dequeue + (num_threads + 1) * batch_size

        if shuffle:
            return tf.train.shuffle_batch(
                self.read_operation,
                batch_size,
                capacity,
                min_after_dequeue,
                num_threads,
                allow_smaller_final_batch=False,
            )
        else:
            return tf.train.batch(
                self.read_operation,
                batch_size,
                num_threads,
                capacity,
                allow_smaller_final_batch=False,
            )
