from os import listdir
from os.path import join, expanduser, isfile

import tensorflow as tf

from dataset.filtering.filters import filtered_filename


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
