import tensorflow as tf

from dataset.batching.tf_record_base import RecordWriter


class ImagePairRecordWriter(RecordWriter):
    def write_image_pair(self, input_file, input_image, input_embedding,
                         target_file, target_image):
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
        self._writer.write(example.SerializeToString())
