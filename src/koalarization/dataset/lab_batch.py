"""Converting to TFRecords (requires Inception checkpoint).

```
$ python -O -m koalarization.dataset.lab_batch <args>
```

Passing `-c path/to/inception_resnet_v2_2016_08_30.ckpt` is highly recommended over passing a url. To download the
checkpoint it separately:

```
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
tar -xvf inception_resnet_v2_2016_08_30.tar.gz
```

Omitting the `-O` (optimize) will print all image names at the moment they are written to a TFRecord. These prints will 
most likely appear all at once, after TensorFlow has written the batch on disk and passes the control back to Python.

Use `-h` to see the available options

"""
import time
from os.path import isdir, join, basename
import argparse

import tensorflow as tf
import tensorflow.contrib.slim as slim

from koalarization.dataset.embedding import (
    prepare_image_for_inception,
    maybe_download_inception,
    inception_resnet_v2,
    inception_resnet_v2_arg_scope,
)
from koalarization.dataset.shared import maybe_create_folder, DIR_RESIZED, DIR_TFRECORD
from koalarization.dataset.shared import progressive_filename_generator
from koalarization.dataset.tfrecords import batch_operations
from koalarization.dataset.tfrecords.images.lab_image_record import LabImageRecordWriter
from koalarization.dataset.tfrecords.images_queue import queue_single_images_from_folder
from koalarization.dataset.embedding.inception_utils import CHECKPOINT_URL


DEFAULT_BATCH_SIZE = 500


class LabImagenetBatcher:
    """Class instance to create TFRecords."""

    def __init__(self, inputs_dir, records_dir, checkpoint_source):
        """Constructor.

        Args:
            inputs_dir (str): Path to folder containing all resized images (input).
            records_dir (str): Path to folder with the TFRecords (output).
            checkpoint_source (str): Set trained inception weights checkpoint, can be the url, the archive or the
                                        file itself.

        Raises:
            Exception: If resized image folder.

        """
        if not isdir(inputs_dir):
            raise Exception("Input folder does not exists: {}".format(inputs_dir))
        self.inputs_dir = inputs_dir

        # Destination folder
        maybe_create_folder(records_dir)
        self.records_dir = records_dir

        # Inception checkpoint
        self.checkpoint_file = maybe_download_inception(checkpoint_source)

        # Utils
        self._examples_count = 0
        self.records_names_gen = progressive_filename_generator(
            join(records_dir, "lab_images_{}.tfrecord")
        )

    def batch_all(self, examples_per_record):
        """.

        Args:
            examples_per_record (int): Batch size.
        """
        operations = self._create_operations(examples_per_record)

        with tf.Session() as sess:
            self._initialize_session(sess)
            self._run_session(sess, operations, examples_per_record)

    def _initialize_session(self, sess):
        """Initialize a new session to run the operations.

        Args:
            sess (tf.Session): Tensorflow session.

        """

        # Initialize the the variables that we introduced (like queues etc.)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Restore the weights from Inception
        # (do not call a global/local variable initializer after this call)
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_file)

    def _create_operations(self, examples_per_record):
        """Create the operations to read images from the queue and extract inception features.

        Args:
            examples_per_record (int): Batch size.

        Returns:
            tuple: Containing all these operations.

        """
        # Create the queue operations
        image_key, image_tensor, _ = queue_single_images_from_folder(self.inputs_dir)

        # Build Inception Resnet v2 operations using the image as input
        # - from rgb to grayscale to loose the color information
        # - from grayscale to rgb just to have 3 identical channels
        # - from a [0, 255] int8 range to [-1,+1] float32
        # - feed the image into inception and get the embedding
        img_for_inception = tf.image.rgb_to_grayscale(image_tensor)
        img_for_inception = tf.image.grayscale_to_rgb(img_for_inception)
        img_for_inception = prepare_image_for_inception(img_for_inception)
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            input_embedding, _ = inception_resnet_v2(
                img_for_inception, is_training=False
            )

        operations = image_key, image_tensor, input_embedding

        return batch_operations(operations, examples_per_record)

    def _run_session(self, sess, operations, examples_per_record):
        """Run the whole reading -> extracting features -> writing to records pipeline in a TensorFlow session.

        Args:
            sess (tf.Session): Tensorflow session.
            operations (tuple): Operations.
            examples_per_record (int): Batch size.

        """
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        self._examples_count = 0

        # These are the only lines where something happens:
        # we execute the operations to get the image, compute the
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
            print(
                "Finished writing {} images in {:.2f}s".format(
                    self._examples_count, time.time() - start_time
                )
            )

        # Wait for threads to finish.
        coord.join(threads)

    def _write_record(self, examples_per_record, operations, sess):
        """Write records.

        Args:
            examples_per_record (int): Batch size.
            operations (tuple): Operations.
            sess (tf.Session): Tensorflow session.
        """
        # The base queue_operation is [a, b, c]
        # The batched queue_operation is [[a1, a2], [b1,b2], [c1, c2]]
        # and not [[a1, b1, c1], [a2, b2, c3]]
        # The result will have the same structure as the batched operations
        results = sess.run(operations)

        # Create a writer to write the images
        with LabImageRecordWriter(next(self.records_names_gen)) as writer:
            # Iterate over each result in the results
            for one_res in zip(*results):
                writer.write_image(*one_res)
                if __debug__:
                    print("Written", basename(one_res[0]))

            self._examples_count += len(results[0])
            print("Record ready:", writer.path)


def _parse_args():
    """Get args.

    Returns:
        Namespace: Get arguments.

    """
    # Argparse setup
    parser = argparse.ArgumentParser(
        description="Takes one folders containing 299x299 images, extracts "
        "the inception resnet v2 features from the image, "
        "serializes the image in Lab space and the embedding and "
        "writes everything in tfrecord files "
        "files in batches on N images"
    )
    parser.add_argument(
        "-i",
        "--inputs-folder",
        default=DIR_RESIZED,
        type=str,
        metavar="FOLDER",
        dest="inputs",
        help="use FOLDER as source of the input images "
        "(default: {}) ".format(DIR_RESIZED),
    ),
    parser.add_argument(
        "-o",
        "--output-folder",
        default=DIR_TFRECORD,
        type=str,
        metavar="FOLDER",
        dest="records",
        help="use FOLDER as destination for the TFRecord "
        "batches (default: {}) ".format(DIR_TFRECORD),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=CHECKPOINT_URL,
        type=str,
        dest="checkpoint",
        help="set the source for the trained inception "
        "weights, can be the url, the archive or the "
        "file itself (default: {}) ".format(CHECKPOINT_URL),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        metavar="N",
        dest="batch_size",
        help="every batch will contain N images, except maybe "
        "the last one (default: {})".format(DEFAULT_BATCH_SIZE),
    )
    args = parser.parse_args()
    return args


# Run from the top folder as:
# python3 -m dataset.lab_batch <args>
if __name__ == "__main__":
    args = _parse_args()
    LabImagenetBatcher(
        inputs_dir=args.inputs,
        records_dir=args.records,
        checkpoint_source=args.checkpoint,
    ).batch_all(examples_per_record=args.batch_size)
