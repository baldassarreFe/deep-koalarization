from os.path import expanduser

import matplotlib
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt

from dataset.shared import dir_metrics
from dataset.tfrecords import ImagePairRecordReader
from unfiltering.unfiltering import define_optimizer
from unfiltering.unfiltering import unfiltering

matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)

    irr = ImagePairRecordReader(
        expanduser('~/imagenet/tfrecords/images_*.tfrecord'))
    read_batched_examples = irr.read_batch(50)

    imgs_in = read_batched_examples['input_image']
    imgs_emb = read_batched_examples['input_embedding']
    imgs_true = read_batched_examples['target_image']

    imgs_delta = unfiltering(imgs_in, imgs_emb)
    imgs_out = imgs_in - imgs_delta
    opt_operations = define_optimizer(imgs_out, imgs_true)

    # Merge all the summaries and set up the writers
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(dir_metrics + '/unfiltering',
                                         sess.graph)

    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(5):
            for batch in range(5):
                print('Epoch:', epoch, 'Batch:', batch, end=' ')
                res = sess.run(opt_operations)
                print('Cost:', res['cost'])
                train_writer.add_summary(sess.run(summaries), 5 * epoch + batch)

        # Evaluation
        res = sess.run({
            'imgs_in': imgs_in,
            'imgs_out': imgs_out,
            'imgs_true': imgs_true
        })

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

    for k in range(len(res['imgs_in'])):
        plt.subplot(1, 3, 1)
        plt.imshow((res['imgs_in'][k] + 1) / 2)
        plt.title('Input (filtered)')
        plt.subplot(1, 3, 2)
        plt.imshow((res['imgs_out'][k] + 1) / 2)
        plt.title('Output (our unfiltered)')
        plt.subplot(1, 3, 3)
        plt.imshow((res['imgs_true'][k] + 1) / 2)
        plt.title('Target (unfiltered)')

        plt.savefig('images/' + str(k) + '.png')
        plt.clf()
        plt.close()
