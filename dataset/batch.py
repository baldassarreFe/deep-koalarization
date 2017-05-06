import time
from os.path import isdir, join

from dataset.batching.images_queue import queue_paired_images_from_folders
from dataset.batching.paired_featured_image_record import ImagePairRecordWriter
from dataset.embedding.inception_resnet_v2 import *
from dataset.embedding.inception_utils import prepare_image_for_inception, \
    inception_resnet_v2_maybe_download
from dataset.filtering.filters import all_filters_with_base_args
from dataset.shared import maybe_create_folder, dir_root


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

    def batch_all(self, examples_per_record):
        # Create the queue operations
        input_key, input_tensor, target_key, target_tensor = \
            queue_paired_images_from_folders(
                self.inputs_dir,
                self.targets_dir,
                [f.__name__ for f in all_filters_with_base_args])

        # Feature extraction operations
        scaled_input_tensor = prepare_image_for_inception(input_tensor)
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2(scaled_input_tensor, is_training=False)

        # Create a writer to write the images
        writer = ImagePairRecordWriter(
            join(self.records_dir, 'images.tfrecord'))

        # Start a new session to run the operations
        with tf.Session() as sess:
            # Initialize the the variables that we introduced (like queues etc.)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Restore the weights from Inception
            # (do not call a global/local variable initializer after this call)
            saver = tf.train.Saver()
            saver.restore(sess, self.checkpoint_file)

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            count = 0
            start_time = time.time()
            
            # These are the only lines where something happens:
            # we execute the operations to get the image pair, compute the
            # embedding and write everything in the TFRecord
            try:
                while not coord.should_stop():
                    in_key, in_img, in_emb, tar_key, tar_img = sess.run(
                        [input_key, input_tensor, end_points['Logits'], target_key,
                         target_tensor])
                    writer.write_image_pair(in_key, in_img, in_emb, tar_key,
                                            tar_img)
                    print('Written', in_key, tar_key)
                    count += 1
            except tf.errors.OutOfRangeError:
                # The string_input_producer queue ran out of strings
                pass
            finally:
                # Ask the threads (filename queue) to stop.
                coord.request_stop()
                print('Finished writing {} pairs in {:.2f}s'
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
