import tensorflow as tf
from keras import backend as K

from colorization import Colorization
from colorization.training_utils import evaluation_pipeline, \
    checkpointing_system, \
    plot_evaluation, training_pipeline, metrics_system, print_log

# PARAMETERS
run_id = 'run1'
epochs = 100
val_number_of_images = 10
total_train_images = 130 * 500
batch_size = 100
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
        print_log('Restoring from: {}'.format(latest_checkpoint), run_id)
        saver.restore(sess, latest_checkpoint)
        print_log(' done!', run_id)
    else:
        print_log('No checkpoint found in: {}'.format(checkpoint_paths), run_id)

    for epoch in range(epochs):
        print_log('Starting epoch: {} (total images {})'
                  .format(epoch, total_train_images), run_id)
        # Training step
        for batch in range(batches):
            print_log('Batch: {}/{}'.format(batch, batches), run_id)
            res = sess.run(opt_operations)
            global_step = res['global_step']
            print_log('Cost: {} Global step: {}'
                      .format(res['cost'], global_step), run_id)
            summary_writer.add_summary(res['summary'], global_step)

        # Save the variables to disk
        save_path = saver.save(sess, checkpoint_paths, global_step)
        print_log("Model saved in: %s" % save_path, run_id)

        # Evaluation step on validation
        res = sess.run(evaluations_ops)
        summary_writer.add_summary(res['summary'], global_step)
        plot_evaluation(res, run_id, epoch)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
