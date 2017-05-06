import itertools
import time
from os.path import isdir, join, basename

from dataset.batching.images_queue import queue_paired_images_from_folders
from dataset.batching.paired_featured_image_record import ImagePairRecordWriter
from dataset.embedding.inception_resnet_v2 import *
from dataset.embedding.inception_utils import prepare_image_for_inception, \
    inception_resnet_v2_maybe_download
from dataset.filtering.filters import all_filters_with_base_args
from dataset.shared import maybe_create_folder, dir_root


def progressive_filename_generator(pattern='file_{}.ext'):
    for i in itertools.count():
        yield pattern.format(i)


class ImagenetBatcher:
    def __init__(self, inputs_dir: str, targets_dir: str, records_dir: str):
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

        self.checkpoint_file = inception_resnet_v2_maybe_download(
            join(dir_root, 'inception_resnet_v2_2016_08_30.ckpt'))

        self.records_names_gen = progressive_filename_generator(
            join(records_dir, 'images_{}.tfrecord'))

    def batch_all(self, examples_per_record):
        operations = self._create_operations()

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

    def _create_operations(self):
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

        return input_key, input_tens, target_key, target_tens, input_embedding

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
        examples_count = 0

        # These are the only lines where something happens:
        # we execute the operations to get the image pair, compute the
        # embedding and write everything in the TFRecord
        try:
            while not coord.should_stop():
                examples_count += self._write_record(examples_per_record,
                                                     operations, sess)
        except tf.errors.OutOfRangeError:
            # The string_input_producer queue ran out of strings
            pass
        finally:
            # Ask the threads (filename queue) to stop.
            coord.request_stop()
            print('Finished writing {} pairs in {:.2f}s'
                  .format(examples_count, time.time() - start_time))

        # Wait for threads to finish.
        coord.join(threads)

    def _write_record(self, examples_per_record, operations, sess):
        examples_written = 0
        # Create a writer to write the images
        with ImagePairRecordWriter(next(self.records_names_gen)) as writer:
            for i in range(examples_per_record):
                results = sess.run(operations)
                writer.write_image_pair(*results)
                print('Written', basename(results[0]),
                      basename(results[2]))
                examples_written += 1
        return examples_written


# Run from the top folder as:
# python3 -m dataset.batch <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_resized, dir_filtered, dir_tfrecord

    # Argparse setup
    # TODO
    parser = argparse.ArgumentParser(
        description='TODO')
    parser.add_argument('-i', '--inputs-folder',
                        default=dir_resized,
                        type=str,
                        metavar='FOLDER',
                        dest='inputs',
                        help='use FOLDER as source of the input images (default: {})'
                        .format(dir_resized)),
    parser.add_argument('-t', '--targets-folder',
                        default=dir_filtered,
                        type=str,
                        metavar='FOLDER',
                        dest='targets',
                        help='use FOLDER as source of the target images (default: {})'
                        .format(dir_filtered))
    parser.add_argument('-o', '--output-folder',
                        default=dir_tfrecord,
                        type=str,
                        metavar='FOLDER',
                        dest='records',
                        help='use FOLDER as destination for the TFRecord batches (default: {})'
                        .format(dir_tfrecord))
    parser.add_argument('-b', '--batch-size',
                        default=20,
                        type=int,
                        metavar='N',
                        dest='batch_size',
                        help='every batch should contain N images (default: 20)')

    args = parser.parse_args()
    ImagenetBatcher(args.inputs, args.targets, args.records) \
        .batch_all(args.batch_size)
