import itertools
import time
from os.path import isdir, join, basename

import tensorflow as tf
import tensorflow.contrib.slim as slim

from dataset.batching import ImagePairRecordWriter
from dataset.batching import queue_paired_images_from_folders, \
    batch_operations
from dataset.embedding import prepare_image_for_inception, \
    maybe_download_inception, inception_resnet_v2, inception_resnet_v2_arg_scope
from dataset.filtering import all_filters_with_base_args
from dataset.shared import maybe_create_folder


def progressive_filename_generator(pattern='file_{}.ext'):
    for i in itertools.count():
        yield pattern.format(i)


class ImagenetBatcher:
    def __init__(self, inputs_dir: str, targets_dir: str,
                 records_dir: str, checkpoint_source: str):
        if not isdir(inputs_dir):
            raise Exception('Input folder does not exists: {}'
                            .format(inputs_dir))
        if not isdir(targets_dir):
            raise Exception('Targets folder does not exists: {}'
                            .format(targets_dir))
        self.inputs_dir = inputs_dir
        self.targets_dir = targets_dir

        # Destination folder
        maybe_create_folder(records_dir)
        self.records_dir = records_dir

        # Inception checkpoint
        self.checkpoint_file = maybe_download_inception(checkpoint_source)

        # Utils
        self._examples_count = 0
        self.records_names_gen = progressive_filename_generator(
            join(records_dir, 'images_{}.tfrecord'))

    def batch_all(self, examples_per_record):
        operations = self._create_operations(examples_per_record)

        with tf.Session() as sess:
            self._initialize_session(sess)
            self._run_session(sess, operations, examples_per_record)

    def _initialize_session(self, sess):
        """
        Initialize a new session to run the operations
        :param sess:
        :return:
        """

        # Initialize the the variables that we introduced (like queues etc.)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Restore the weights from Inception
        # (do not call a global/local variable initializer after this call)
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_file)

    def _create_operations(self, examples_per_record):
        """
        Create the operations to read images from the queue and
        extract inception features
        :return: a tuple containing all these operations
        """
        # Create the queue operations
        input_key, input_tens, target_key, target_tens = \
            queue_paired_images_from_folders(
                self.inputs_dir,
                self.targets_dir,
                [f.__name__ for f in all_filters_with_base_args])

        # Build Inception Resnet v2 operations using the image as input
        scaled_input_tensor = prepare_image_for_inception(input_tens)
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            input_embedding, _ = inception_resnet_v2(scaled_input_tensor,
                                                     is_training=False)

        operations = input_key, input_tens, target_key, target_tens, input_embedding

        return batch_operations(operations, examples_per_record)

    def _run_session(self, sess, operations, examples_per_record):
        """
        Run the whole reading -> extracting features -> writing to records
        pipeline in a TensorFlow session
        :param sess:
        :param operations:
        :param examples_per_record:
        :return:
        """

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        self._examples_count = 0

        # These are the only lines where something happens:
        # we execute the operations to get the image pair, compute the
        # embedding and write everything in the TFRecord
        try:
            while not coord.should_stop():
                self._write_record(examples_per_record, operations, sess)
        except tf.errors.OutOfRangeError:
            # The string_input_producer queue ran out of strings
            pass
        finally:
            # Ask the threads (filename queue) to stop.
            coord.request_stop()
            print('Finished writing {} pairs in {:.2f}s'
                  .format(self._examples_count, time.time() - start_time))

        # Wait for threads to finish.
        coord.join(threads)

    def _write_record(self, examples_per_record, operations, sess):
        # The base queue_operation is [a, b, c]
        # The batched queue_operation is [[a1, a2], [b1,b2], [c1, c2]]
        # and not [[a1, b1, c1], [a2, b2, c3]]
        # The result will have the same structure as the batched operations
        results = sess.run(operations)

        # Create a writer to write the images
        with ImagePairRecordWriter(next(self.records_names_gen)) as writer:
            # Iterate over each result in the results
            for one_res in zip(*results):
                writer.write_image_pair(*one_res)
                if __debug__:
                    print('Written', basename(one_res[0]), basename(one_res[2]))

            self._examples_count += len(results[0])
            print('Record ready:', writer.path)


# Run from the top folder as:
# python3 -m dataset.batch <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_resized, dir_filtered, dir_tfrecord
    from dataset.embedding.inception_utils import checkpoint_url

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Takes two folders containing paired images (resized and '
                    'filtered_resized), extracts the inception resnet v2 '
                    'features from the filtered image, serializes the images '
                    'and the embedding and writes everything in tfrecord '
                    'files in batches on N images')
    parser.add_argument('-i', '--inputs-folder',
                        default=dir_resized,
                        type=str,
                        metavar='FOLDER',
                        dest='inputs',
                        help='use FOLDER as source of the input images '
                             '(default: {}) '
                        .format(dir_resized)),
    parser.add_argument('-t', '--targets-folder',
                        default=dir_filtered,
                        type=str,
                        metavar='FOLDER',
                        dest='targets',
                        help='use FOLDER as source of the target images '
                             '(default: {}) '
                        .format(dir_filtered))
    parser.add_argument('-o', '--output-folder',
                        default=dir_tfrecord,
                        type=str,
                        metavar='FOLDER',
                        dest='records',
                        help='use FOLDER as destination for the TFRecord '
                             'batches (default: {}) '
                        .format(dir_tfrecord))
    parser.add_argument('-c', '--checkpoint',
                        default=checkpoint_url,
                        type=str,
                        dest='checkpoint',
                        help='set the source for the trained inception '
                             'weights, can be the url, the archive or the '
                             'file itself (default: {}) '
                        .format(checkpoint_url))
    parser.add_argument('-b', '--batch-size',
                        default=500,
                        type=int,
                        metavar='N',
                        dest='batch_size',
                        help='every batch will contain N images, except maybe '
                             'the last one (default: 500)')

    args = parser.parse_args()
    ImagenetBatcher(args.inputs, args.targets, args.records, args.checkpoint) \
        .batch_all(args.batch_size)
