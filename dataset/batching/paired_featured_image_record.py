import tensorflow as tf

from dataset.batching.tf_record_base import RecordWriter, RecordReader


class ImagePairRecordWriter(RecordWriter):
    def write_image_pair(self, input_file, input_image, target_file,
                         target_image, input_embedding):
        """
        Writes into this TFRecord an Example representing a pair of images,
        including the image names, the image data and and embedding of the
        input image
        :param input_file:
        :param input_image:
        :param input_embedding:
        :param target_file:
        :param target_image:
        """
        example = tf.train.Example(features=tf.train.Features(feature={
            'input_file': self._bytes_feature(input_file),
            'input_image': self._bytes_feature(input_image.tostring()),
            'input_embedding': self._bytes_feature(input_embedding.tostring()),
            'target_file': self._bytes_feature(target_file),
            'target_image': self._bytes_feature(target_image.tostring())
        }))
        self.write(example.SerializeToString())


class ImageRecordReader(RecordReader):
    def _create_read_operation(self):
        features = tf.parse_single_example(
            self._tfrecord_serialized,
            features={
                'input_file': tf.FixedLenFeature([], tf.string),
                'input_image': tf.FixedLenFeature([], tf.string),
                'input_embedding': tf.FixedLenFeature([], tf.string),
                'target_file': tf.FixedLenFeature([], tf.string),
                'target_image': tf.FixedLenFeature([], tf.string),
            })
        shape = [299, 299, 3]

        # Filenames
        input_file = features['input_file']
        target_file = features['target_file']

        # Images
        input_image = tf.decode_raw(features['input_image'], tf.uint8)
        input_image = tf.reshape(input_image, shape)
        input_image = tf.cast(input_image, tf.float32) * (1. / 255)
        target_image = tf.decode_raw(features['target_image'], tf.uint8)
        target_image = tf.reshape(target_image, shape)
        target_image = tf.cast(target_image, tf.float32) * (1. / 255)

        # Embeddings
        input_embedding = tf.decode_raw(features['target_image'], tf.float32)

        return [input_file, input_image, input_embedding, target_file,
                target_image]
