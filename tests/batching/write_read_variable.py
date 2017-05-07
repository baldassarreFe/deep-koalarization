"""
Write some variable size examples in a tfrecord, then read them one at the time
(all the tensors have differente shapes, so we can't batch them)

Run from the top folder as:
python3 -m tests.batching.write_read_variable
"""
import tensorflow as tf

from dataset.shared import dir_tfrecord
from tests.tfrecords import VariableSizeTypesRecordWriter, \
    VariableSizeTypesRecordReader

# WRITING
with VariableSizeTypesRecordWriter('variable.tfrecord', dir_tfrecord) as writer:
    for i in range(2):
        writer.write_test()

# READING
reader = VariableSizeTypesRecordReader('variable.tfrecord', dir_tfrecord)
read_one_example = reader.read_one()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Coordinate the queue of tfrecord files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Reading examples sequentially one by one
    for j in range(3):
        fetches = sess.run(read_one_example)
        print('Read:', fetches)

    # Finish off the queue coordinator.
    coord.request_stop()
    coord.join(threads)
