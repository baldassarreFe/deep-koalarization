import tensorflow as tf

from ..base import BatchableRecordReader, RecordWriter


class SingleImageRecordWriter(RecordWriter):
    def __init__(self, tfrecord_name, dest_folder="", img_shape=(299, 299, 3)):
        super().__init__(tfrecord_name, dest_folder)
        self.img_shape = img_shape

    def write_image(self, key, img):
        assert img.shape == self.img_shape
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "key": self._bytes_feature(key),
                    "image": self._bytes_feature(img.tobytes()),
                }
            )
        )
        self.write(example.SerializeToString())


class SingleImageRecordReader(BatchableRecordReader):
    def __init__(self, tfrecord_name, dest_folder="", img_shape=(299, 299, 3)):
        super().__init__(tfrecord_name, dest_folder)
        self.img_shape = img_shape

    def _create_read_operation(self):
        features = tf.parse_single_example(
            self._tfrecord_serialized,
            features={
                "key": tf.FixedLenFeature([], tf.string),
                "image": tf.FixedLenFeature([], tf.string),
            },
        )

        image = tf.decode_raw(features["image"], tf.uint8)
        image = tf.reshape(image, shape=self.img_shape)
        image = tf.cast(image, tf.float32) * (1.0 / 255)

        return {"key": features["key"], "image": image}
