import tensorflow as tf

from dataset.tfrecords import BatchableRecordReader, RecordWriter


class ImagePairRecordWriter(RecordWriter):
    def __init__(self, tfrecord_name, dest_folder='',
                 img_shape=(299, 299, 3), embedding_size=1001):
        super().__init__(tfrecord_name, dest_folder)
        self.img_shape = img_shape
        self.embedding_size = embedding_size

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
        assert input_image.shape == self.img_shape
        assert target_image.shape == self.img_shape
        assert input_embedding.size == self.embedding_size
        example = tf.train.Example(features=tf.train.Features(feature={
            'input_file': self._bytes_feature(input_file),
            'input_image': self._bytes_feature(input_image.tostring()),
            'input_embedding': self._float32_list(input_embedding.flatten()),
            'target_file': self._bytes_feature(target_file),
            'target_image': self._bytes_feature(target_image.tostring())
        }))
        self.write(example.SerializeToString())


class ImagePairRecordReader(BatchableRecordReader):
    def __init__(self, tfrecord_name, dest_folder='',
                 img_shape=(299, 299, 3), embedding_size=1001):
        super().__init__(tfrecord_name, dest_folder)
        self.img_shape = img_shape
        self.embedding_size = embedding_size

    def _create_read_operation(self):
        features = tf.parse_single_example(
            self._tfrecord_serialized,
            features={
                'input_file': tf.FixedLenFeature([], tf.string),
                'input_image': tf.FixedLenFeature([], tf.string),
                'input_embedding': tf.FixedLenFeature(
                    [self.embedding_size], tf.float32),
                'target_file': tf.FixedLenFeature([], tf.string),
                'target_image': tf.FixedLenFeature([], tf.string),
            })

        # Filenames
        input_file = features['input_file']
        target_file = features['target_file']

        # Images
        input_image = tf.decode_raw(features['input_image'], tf.uint8)
        input_image = tf.reshape(input_image, self.img_shape)
        input_image = tf.cast(input_image, tf.float32) * (1. / 255)
        target_image = tf.decode_raw(features['target_image'], tf.uint8)
        target_image = tf.reshape(target_image, self.img_shape)
        target_image = tf.cast(target_image, tf.float32) * (1. / 255)

        # Embeddings
        input_embedding = features['input_embedding']

        return {
            'input_file': input_file,
            'input_image': input_image,
            'input_embedding': input_embedding,
            'target_file': target_file,
            'target_image': target_image
        }
