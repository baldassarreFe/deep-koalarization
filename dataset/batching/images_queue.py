import multiprocessing
from os import listdir
from os.path import join, expanduser, isfile

import tensorflow as tf

from dataset.filtering import filtered_filename


def queue_single_images_from_folder(folder):
    # Normalize the path
    folder = expanduser(folder)

    # This queue will yield a filename every time it is polled
    file_matcher = tf.train.match_filenames_once(join(folder, '*.jpeg'))

    # NOTE: if num_epochs is set to something different than None, then we
    # need to run tf.local_variables_initializer when launching the session!!
    # https://www.tensorflow.org/api_docs/python/tf/train/string_input_producer
    filename_queue = tf.train.string_input_producer(
        file_matcher, shuffle=False, num_epochs=1)

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


def image_pair_paths_generator(inputs_folder, target_folder, suffixes):
    for input_file in listdir(inputs_folder):
        input_path = join(inputs_folder, input_file)
        if isfile(input_path):
            for suff in suffixes:
                target_file = filtered_filename(input_file, suff)
                target_path = join(target_folder, target_file)
                if isfile(target_path):
                    yield input_path, target_path


def queue_paired_images_from_folders(inputs_folder, targets_folder, suffixes):
    """
    If the suffixes given are `['one', 'two']`
    The expected folder structure is:

    inputs
    ├── aaa.jpeg
    └── bbb.jpeg

    targets
    ├── aaa_one.jpeg
    ├── aaa_two.jpeg
    ├── bbb_one.jpeg
    └── bbb_two.jpeg

    :param inputs_folder:
    :param targets_folder:
    :param suffixes:
    :return:
    """
    # TODO this docstring needs formatting

    # Normalize paths
    inputs_folder = expanduser(inputs_folder)
    targets_folder = expanduser(targets_folder)

    # Create two lists with the matching files in the corresponding positions
    inputs_paths, targets_paths = zip(*image_pair_paths_generator(
        inputs_folder, targets_folder, suffixes))

    # Create two queues from the lists
    inputs_queue = tf.train.string_input_producer(inputs_paths, shuffle=False,
                                                  num_epochs=1)
    targets_queue = tf.train.string_input_producer(targets_paths, shuffle=False,
                                                   num_epochs=1)

    # Read paired images from the two queues
    image_reader = tf.WholeFileReader()
    input_key, input_file = image_reader.read(inputs_queue)
    target_key, target_file = image_reader.read(targets_queue)

    # The file needs to be decoded as image and we also need its dimensions
    input_tensor = tf.image.decode_jpeg(input_file)
    target_tensor = tf.image.decode_jpeg(target_file)

    # Note: nothing has happened yet, we've only defined operations,
    # what we return are tensors
    return input_key, input_tensor, target_key, target_tensor


def batch_operations(operations, batch_size):
    """
    Once you have created the operation(s) with the other methods of this class,
    use this method to batch it(them).

    :Note:

        If a single queue operation is `[a, b, c]`,
        the batched queue_operation will be `[[a1, a2], [b1,b2], [c1, c2]]`
        and not `[[a1, b1, c1], [a2, b2, c3]]`

    :param operations: can be a tensor or a list of tensors
    :param batch_size: the batch
    :return:
    """
    # The internet gave me these numbers
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
