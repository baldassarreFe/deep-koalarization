import tensorflow as tf

from dataset.batching.tf_record_base import RecordWriter


class ImagePairRecordWriter(RecordWriter):
    def write_image_pair(self, key_one, img_one, key_two, img_two):
        example = tf.train.Example(features=tf.train.Features(feature={
            'key_one': self._bytes_feature(key_one),
            'key_two': self._bytes_feature(key_two),
            'image_one': self._bytes_feature(img_one.tostring()),
            'image_two': self._bytes_feature(img_two.tostring())
        }))
        self._writer.write(example.SerializeToString())
