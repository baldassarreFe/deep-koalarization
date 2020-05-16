import multiprocessing
from os.path import join, expanduser

import tensorflow as tf


def queue_single_images_from_folder(folder):
    """Create queue of images.

    Args:
        folder (str): Folder.

    """
    # Normalize the path
    folder = expanduser(folder)

    # This queue will yield a filename every time it is polled
    file_matcher = tf.train.match_filenames_once(join(folder, "*.jpeg"))

    # NOTE: if num_epochs is set to something different than None, then we
    # need to run tf.local_variables_initializer when launching the session!!
    # https://www.tensorflow.org/api_docs/python/tf/train/string_input_producer
    filename_queue = tf.train.string_input_producer(
        file_matcher, shuffle=False, num_epochs=1
    )

    # This is the reader we'll use to read each image given the file name
    image_reader = tf.WholeFileReader()

    # This operation polls the queue and reads the image
    image_key, image_file = image_reader.read(filename_queue)

    # The file needs to be decoded as image and we also need its dimensions
    image_tensor = tf.image.decode_jpeg(image_file)
    image_shape = tf.shape(image_tensor)

    # Note: nothing has happened yet, we've only defined operations,
    # what we return are tensors
    return image_key, image_tensor, image_shape


def batch_operations(operations, batch_size):
    """Once you have created the operation(s) with the other methods of this class,
    use this method to batch it (them).

    Args:
        operations: Can be a tensor or a list of tensors.
        batch_size (int): Batch

    Returns:
        [type]: [description]
    """
    # Recommended configuration for these parameters (found online)
    num_threads = multiprocessing.cpu_count()
    min_after_dequeue = 3 * batch_size
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    return tf.train.batch(
        operations,
        batch_size,
        num_threads,
        capacity,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
    )
