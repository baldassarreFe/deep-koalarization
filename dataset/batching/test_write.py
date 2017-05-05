import time

import tensorflow as tf

from dataset.batching.images_queue import queue_single_images_from_folder
from dataset.batching.paired_image_record import ImagePairRecordWriter
from dataset.batching.single_image_record import ImageRecordWriter

# Just for testing, reads images from a folder two at the time and writes
# them in single and pairs

# Create the queue operations
img_key, img_tensor, _ = queue_single_images_from_folder('~/imagenet/resized')

# Create a writer to write_image the images
single_writer = ImageRecordWriter('single_images.tfrecord')
pair_writer = ImagePairRecordWriter('paired_images.tfrecord')

# Start a new session to run the operations
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    count = 0
    start_time = time.time()
    try:
        while not coord.should_stop():
            key_one, img_one = sess.run([img_key, img_tensor])
            key_two, img_two = sess.run([img_key, img_tensor])

            single_writer.write_image(key_one, img_one)
            single_writer.write_image(key_two, img_two)

            pair_writer.write_image_pair(key_one, img_one, key_two, img_two)

            print('Written:\n- {}\n- {}'.format(key_one, key_two))
            count += 2
    except tf.errors.OutOfRangeError:
        # It's all right, it's just the string_input_producer queue telling
        # us that it has run out of strings
        pass
    finally:
        # Ask the threads (filename queue) to stop.
        coord.request_stop()
        print('Finished writing {} files in {}'
              .format(count, time.time() - start_time))

    # Wait for threads to finish.
    coord.join(threads)

single_writer.close()
pair_writer.close()
