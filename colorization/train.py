from keras import backend as K

from colorization import Colorization
from colorization.training_utils import evaluation_pipeline, \
    checkpointing_system, \
    plot_evaluation, training_pipeline, metrics_system, print_log, print_term

import tensorflow as tf


# PARAMETERS
run_id = 'run1'
epochs = 100  #default 100
val_number_of_images = 10
total_train_images = 75000  #default 130 * 500
batch_size = 32  #default 100
learning_rate = 0.001
batches = total_train_images // batch_size

# START
print_term('Starting session...', run_id)
sess = tf.Session()
K.set_session(sess)
print_term('Started session...', run_id)

# Build the network and the various operations
print_term('Building network...', run_id)
col = Colorization(256)

opt_operations = training_pipeline(col, learning_rate, batch_size)
evaluations_ops = evaluation_pipeline(col, val_number_of_images)
summary_writer = metrics_system(run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)
print_term('Built network', run_id)

with sess.as_default():
    # tf.summary.merge_all()
    # writer = tf.summary.FileWriter('./graphs', sess.graph)

    # Initialize
    print_term('Initializing variables...', run_id)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    print_term('Initialized variables', run_id)

    # Coordinate the loading of image files.
    print_term('Coordinating loaded image files...', run_id)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print_term('Coordinate loaded image files', run_id)

    # Restore
    if latest_checkpoint is not None:
        print_term('Restoring from: {}'.format(latest_checkpoint), run_id)
        saver.restore(sess, latest_checkpoint)
        print_term(' done!', run_id)
    else:
        print_term('No checkpoint found in: {}'.format(checkpoint_paths), run_id)

    # Actual training with epochs as iteration
    for epoch in range(epochs):
        print_term('Starting epoch: {} (total images {})'
                  .format(epoch, total_train_images), run_id)
        # Training step
        for batch in range(batches):
            print_term('Batch: {}/{}'.format(batch, batches), run_id)
            res = sess.run(opt_operations)
            global_step = res['global_step']
            print_term('Cost: {} Global step: {}'
                      .format(res['cost'], global_step), run_id, res['cost'])
            summary_writer.add_summary(res['summary'], global_step)

        # Save the variables to disk
        save_path = saver.save(sess, checkpoint_paths, global_step)
        print_term("Model saved in: %s" % save_path, run_id)

        # Evaluation step on validation
        res = sess.run(evaluations_ops)
        summary_writer.add_summary(res['summary'], global_step)
        plot_evaluation(res, run_id, epoch)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
