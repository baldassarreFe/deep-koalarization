import numpy as np
import tensorflow as tf

from dataset.batching.tf_record_base import RecordWriter, RecordReader


class ImageRecordWriter(RecordWriter):
    def write_image(self, key, img):
        example = tf.train.Example(features=tf.train.Features(feature={
            'key': self._bytes_feature(key),
            'image': self._bytes_feature(img.tostring()),
            'shape': self._bytes_feature(np.array(img.shape).tostring()),
            # 'height': self._int64_feature(img.shape[0]),
            # 'width': self._int64_feature(img.shape[1]),
            # 'depth': self._int64_feature(img.shape[2]),
        }))
        self.write(example.SerializeToString())


class ImageRecordReader(RecordReader):
    def _create_read_operation(self):
        features = tf.parse_single_example(
            self._tfrecord_serialized,
            features={
                'key': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string),
                # 'shape': tf.FixedLenFeature([], tf.string)
                # 'height': tf.FixedLenFeature([], tf.int64),
                # 'width': tf.FixedLenFeature([], tf.int64),
                # 'depth': tf.FixedLenFeature([], tf.int64),
            })

        # height = features['height']
        # width = features['width']
        # depth = features['depth']

        # The reshape of image won't work if shape is a int64,
        # so we need to cast to int32 first
        # shape = tf.decode_raw(features['shape'], tf.int64)
        # shape = tf.cast(shape, tf.int32)
        shape = [299, 299, 3]

        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, shape)
        image = tf.cast(image, tf.float32) * (1. / 255)
        key = features['key']

        return [key, image]
