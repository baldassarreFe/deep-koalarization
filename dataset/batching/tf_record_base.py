import tensorflow as tf


class RecordWriter:
    """
    Wrapper around a TFRecordWriter that adds the methods to serialize an
    image and its shape
    """

    def __init__(self, tfrecord_name):
        self._writer = tf.python_io.TFRecordWriter(tfrecord_name)

    def close(self):
        self._writer.close()

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
