import os

import tensorflow as tf
from shared import *

filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(os.path.join(dir_resized, '*.jpeg')))

image_reader = tf.WholeFileReader()

file_name, image_file = image_reader.read(filename_queue)
image = tf.image.decode_jpeg(image_file)

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    sess.run(tf.global_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_shape = tf.shape(image)
    image_tensor = sess.run([image_shape])
    print(image_tensor)

    image_tensor = sess.run([image_shape])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)