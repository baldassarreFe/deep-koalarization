import numpy as np
import tensorflow as tf

from dataset.batching.tf_record_base import RecordWriter


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
        self._writer.write(example.SerializeToString())


class ImageRecordReader:
    def __init__(self, tfrecord_name):
        filename_queue = tf.train.string_input_producer([tfrecord_name])
        reader = tf.TFRecordReader()
        tfrecord_key, tfrecord_serialized = reader.read(filename_queue)

        features = tf.parse_single_example(
            tfrecord_serialized,
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
        self._read_operation = [key, image]

    def read_one(self):
        return self._read_operation

    def read_batch(self, batch_size):
        num_threads = 1
        min_after_dequeue = 10 * batch_size
        capacity = min_after_dequeue + (num_threads + 1) * batch_size
        key_batch, image_batch = tf.train.shuffle_batch(
            self._read_operation,
            batch_size, capacity,
            min_after_dequeue, num_threads,
            allow_smaller_final_batch=True)
        return key_batch, image_batch
