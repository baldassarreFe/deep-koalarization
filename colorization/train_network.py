from os.path import expanduser, join

import matplotlib

matplotlib.use('Agg')

import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt
from tensorflow.python.training.saver import latest_checkpoint

from colorization import colorization, define_optimizer, l_to_rgb, lab_to_rgb
from dataset.tfrecords import LabImageRecordReader
from dataset.shared import dir_metrics, dir_tfrecord, dir_checkpoints

matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)

if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)

    irr = LabImageRecordReader('lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(50)

    imgs_l = read_batched_examples['image_l']
    imgs_true_ab = read_batched_examples['image_ab']
    imgs_emb = read_batched_examples['image_embedding']

    # Build the network and the optimizer step
    imgs_ab = colorization(imgs_l, imgs_emb)
    opt_operations = define_optimizer(imgs_ab, imgs_true_ab)

    # Merge all the summaries and set up the writers
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(dir_metrics, sess.graph)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    checkpoint_paths = join(dir_checkpoints, 'colorization')
    latest_checkpoint = latest_checkpoint(dir_checkpoints)

    with sess.as_default():
        # Initialize
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Restore
        if latest_checkpoint is not None:
            saver.restore(sess, latest_checkpoint)

        for epoch in range(5):
            for batch in range(3):
                print('Epoch:', epoch, 'Batch:', batch, end=' ')
                res = sess.run(opt_operations)
                print('Cost:', res['cost'])
                train_writer.add_summary(sess.run(summaries), 5 * epoch + batch)

            # Save the variables to disk
            save_path = saver.save(sess, checkpoint_paths, global_step=epoch)
            print("Model saved in file: %s" % save_path)

            # Evaluation
            res = sess.run({
                'imgs_l': imgs_l,
                'imgs_ab': imgs_ab,
                'imgs_true_ab': imgs_true_ab
            })

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

                plt.savefig(join(expanduser('~/imagenet/colorized'),
                                 '{}_{}.png'.format(epoch, k)))
                plt.clf()
                plt.close()

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)
