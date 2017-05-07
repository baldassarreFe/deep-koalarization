"""
Read and show single images from a tfrecord

Run from the top folder as:
python3 -m tests.batching.single_image_read
"""
from os.path import basename

import matplotlib.pyplot as plt
import tensorflow as tf

from dataset.shared import dir_tfrecord
from tests.tfrecords.single_image_record import SingleImageRecordReader

# Important: read_batch MUST be called before start_queue_runners,
# otherwise the internal shuffle queue gets created but its
# threads won't start
irr = SingleImageRecordReader('single_images.tfrecord', dir_tfrecord)
read_one_example = irr.read_one()
read_batched_examples = irr.read_batch(10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Reading images sequentially one by one
    for i in range(6):
        res = sess.run(read_one_example)
        plt.subplot(2, 3, i + 1)
        plt.imshow(res['image'])
        plt.axis('off')
        print('Read', basename(res['key']))
    plt.show()

    # Reading images in batch
    res = sess.run(read_batched_examples)
    print(res['key'], res['image'].shape)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
