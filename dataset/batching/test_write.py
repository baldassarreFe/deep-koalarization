import time

import tensorflow as tf

from dataset.batching.images_queue import queue_single_images_from_folder
from dataset.batching.single_image_record import ImageRecordWriter
from dataset.shared import dir_resized, dir_tfrecord

# Just for testing: read images from a folder write them in a record

# Create the queue operations
img_key, img_tensor, _ = queue_single_images_from_folder(dir_resized)

# Create a writer to write_image the images
single_writer = ImageRecordWriter('single_images.tfrecord', dir_tfrecord)

# Start a new session to run the operations
with tf.Session() as sess:
    sess.run(
        [tf.global_variables_initializer(), tf.local_variables_initializer()])

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    count = 0
    start_time = time.time()
    try:
        while not coord.should_stop():
            key, img = sess.run([img_key, img_tensor])
            single_writer.write_image(key, img)
            print('Written: {}'.format(key))
            count += 2
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

single_writer.close()
