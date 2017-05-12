import tensorflow as tf
from keras import backend as K

from colorization import Colorization
from colorization.training_utils import evaluation_pipeline, \
    checkpointing_system, \
    plot_evaluation, training_pipeline, metrics_system

# PARAMETERS
run_id = 'run{}'.format(1)
epochs = 1
val_number_of_images = 20
total_train_images = 10
batch_size = 10
learning_rate = 0.001
batches = total_train_images // batch_size

# START
sess = tf.Session()
K.set_session(sess)

# Build the network and the various operations
col = Colorization(256)
opt_operations = training_pipeline(col, learning_rate, batch_size)
evaluations_ops = evaluation_pipeline(col, val_number_of_images)
summary_writer = metrics_system(run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)

with sess.as_default():
    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Restore
    if latest_checkpoint is not None:
        print('Restoring from:', latest_checkpoint, end='')
        saver.restore(sess, latest_checkpoint)
        print('done!')
    else:
        print('No checkpoint found in:', checkpoint_paths)

    for epoch in range(epochs):
        print('Starting epoch: {} (total images {})'
              .format(epoch, total_train_images))
        # Training step
        for batch in range(batches):
            print(' Batch: {}/{}'.format(batch, batches), end=' ')
            res = sess.run(opt_operations)
            global_step = res['global_step']
            print('Cost:', res['cost'], 'Global step', global_step, )
            summary_writer.add_summary(res['summary'], global_step)

        # Save the variables to disk
        save_path = saver.save(sess, checkpoint_paths, global_step)
        print("Model saved in: %s" % save_path)

        # Evaluation step on validation
        res = sess.run(evaluations_ops)
        summary_writer.add_summary(res['summary'], global_step)
        plot_evaluation(res, run_id, epoch)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
