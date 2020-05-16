import numpy as np
import tensorflow as tf
from skimage import color, transform

from ..base import BatchableRecordReader, RecordWriter

width = 224
height = 224
depth = 3
img_shape = (width, height, depth)
embedding_size = 1001


class LabImageRecordWriter(RecordWriter):
    img_shape = img_shape
    embedding_size = embedding_size

    def write_image(self, img_file, image, img_embedding):
        img = transform.resize(image, img_shape, mode="constant")
        lab = color.rgb2lab(img).astype(np.float32)
        l_channel = 2 * lab[:, :, 0] / 100 - 1
        ab_channels = lab[:, :, 1:] / 127
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image_name": self._bytes_feature(img_file),
                    "image_l": self._float32_list(l_channel.flatten()),
                    "image_ab": self._float32_list(ab_channels.flatten()),
                    "image_embedding": self._float32_list(img_embedding.flatten()),
                }
            )
        )
        self.write(example.SerializeToString())


class LabImageRecordReader(BatchableRecordReader):
    img_shape = img_shape
    embedding_size = embedding_size

    def _create_read_operation(self):
        features = tf.parse_single_example(
            self._tfrecord_serialized,
            features={
                "image_name": tf.FixedLenFeature([], tf.string),
                "image_l": tf.FixedLenFeature([width * height], tf.float32),
                "image_ab": tf.FixedLenFeature([width * height * 2], tf.float32),
                "image_embedding": tf.FixedLenFeature([embedding_size], tf.float32),
            },
        )

        image_l = tf.reshape(features["image_l"], shape=[width, height, 1])
        image_ab = tf.reshape(features["image_ab"], shape=[width, height, 2])

        return {
            "image_name": features["image_name"],
            "image_l": image_l,
            "image_ab": image_ab,
            "image_embedding": features["image_embedding"],
        }
