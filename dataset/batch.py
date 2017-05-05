import time
from os.path import isdir, join

import tensorflow as tf

from dataset.batching.images_queue import queue_paired_images_from_folders
from dataset.batching.paired_image_record import ImagePairRecordWriter
from dataset.filtering.filters import all_filters_with_base_args
from dataset.shared import maybe_create_folder


class ImagenetBatcher:
    def __init__(self, inputs_dir: str, targets_dir, records_dir: str):
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

    def batch_all(self, batch_size):
        # Create the queue operations
        suffixes = map(lambda f: f.__name__, all_filters_with_base_args)
        input_key, input_tensor, target_key, target_tensor = \
            queue_paired_images_from_folders('~/imagenet/resized',
                                             '~/imagenet/filtered', suffixes)

        # TODO add feature extraction here from input_tensor

        # Create a writer to write_image the images
        writer = ImagePairRecordWriter(
            join(self.records_dir, 'images.tfrecord'))

        # Start a new session to run the operations
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            count = 0
            start_time = time.time()

            # These are the only lines where something happens:
            # we execute the operations to get one image and print everything.
            try:
                while not coord.should_stop():
                    in_key, in_img, tar_key, tar_img = sess.run(
                        [input_key, input_tensor, target_key, target_tensor])
                    writer.write_image_pair(in_key, in_img, tar_key, tar_img)
                    print('Written', in_key, tar_key)
                    count += 1
            except tf.errors.OutOfRangeError:
                # It's all right, it's just the string_input_producer queue telling
                # us that it has run out of strings
                pass
            finally:
                # Ask the threads (filename queue) to stop.
                coord.request_stop()
                print('Finished reading {} files in {}'
                      .format(count, time.time() - start_time))

            # Wait for threads to finish.
            coord.join(threads)


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
