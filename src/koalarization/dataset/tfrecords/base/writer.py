from os.path import join

import tensorflow as tf

compression = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)


class RecordWriter(tf.python_io.TFRecordWriter):
    """
    A commodity subclass of TFRecordWriter that adds the methods to
    easily serialize different data types.
    """

    def __init__(self, tfrecord_name, dest_folder=""):
        self.path = join(dest_folder, tfrecord_name)
        super().__init__(self.path, options=compression)

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64(single_int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[single_int]))

    @staticmethod
    def _int64_list(list_of_int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_int))

    @staticmethod
    def _float32(single_float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[single_float]))

    @staticmethod
    def _float32_list(list_of_floats):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))
