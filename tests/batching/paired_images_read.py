"""
Reads and show paired images from a tfrecord

Run from the top folder as:
python3 -m dataset.batching.paired_images_read
"""
from os.path import basename

import matplotlib.pyplot as plt
import tensorflow as tf

from dataset.batching.paired_featured_image_record import ImagePairRecordReader
from dataset.shared import dir_tfrecord

# Important: read_batch MUST be called before start_queue_runners,
# otherwise the internal shuffle queue gets created but its
# threads won't start
irr = ImagePairRecordReader('images_0.tfrecord', dir_tfrecord)
read_one_example = irr.read_one()
read_batched_examples = irr.read_batch(10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Reading images sequentially one by one
    for i in range(0, 8, 2):
        res = sess.run(read_one_example)
        plt.subplot(2, 4, i + 1)
        plt.imshow(res['input_image'])
        plt.axis('off')
        plt.subplot(2, 4, i + 2)
        plt.imshow(res['target_image'])
        plt.axis('off')
        print('Read:',
              '\n\tinput', basename(res['input_file']),
              '\n\ttarget', basename(res['target_file']),
              '\n\tembedding', res['input_embedding'])
    plt.show()

    # Reading images in batch
    res = sess.run(read_batched_examples)
    print(res['input_image'].shape,
          res['target_image'].shape,
          res['input_embedding'].shape)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
