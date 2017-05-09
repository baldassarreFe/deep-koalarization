from os.path import expanduser, join

import matplotlib
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt

from colorization import colorization, define_optimizer, l_to_rgb, lab_to_rgb
from dataset.shared import dir_metrics
from dataset.tfrecords import LabImageRecordReader

matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)

    irr = LabImageRecordReader(
        expanduser('~/imagenet/tfrecords/lab_images_0.tfrecord'))
    read_batched_examples = irr.read_batch(10)

    imgs_l = read_batched_examples['image_l']
    imgs_true_ab = read_batched_examples['image_ab']
    imgs_emb = read_batched_examples['image_embedding']

    # Build the network and the optimizer step
    imgs_ab = colorization(imgs_l, imgs_emb)
    opt_operations = define_optimizer(imgs_ab, imgs_true_ab)

    # Merge all the summaries and set up the writers
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(dir_metrics, sess.graph)

    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(1):
            for batch in range(1):
                print('Epoch:', epoch, 'Batch:', batch, end=' ')
                res = sess.run(opt_operations)
                print('Cost:', res['cost'])
                train_writer.add_summary(sess.run(summaries), 5 * epoch + batch)

        # Evaluation
        res = sess.run({
            'imgs_l': imgs_l,
            'imgs_ab': imgs_ab,
            'imgs_true_ab': imgs_true_ab
        })

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

    for k in range(len(res['imgs_l'])):
        img_gray = l_to_rgb(res['imgs_l'][k][:, :, 0])
        img_output = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                                res['imgs_ab'][k])
        img_true = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                              res['imgs_true_ab'][k])

        plt.subplot(1, 3, 1)
        plt.imshow(img_gray)
        plt.title('Input (grayscale)')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_output)
        plt.title('Network output')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(img_true)
        plt.title('Target (original)')
        plt.axis('off')

        plt.savefig(join(expanduser('~/imagenet/colorized'), str(k) + '.png'))
        plt.clf()
        plt.close()
