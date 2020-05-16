import argparse
from pathlib import Path

import tensorflow as tf
from keras import backend as K

from .network_definition import Colorization
from .training_utils import (
    evaluation_pipeline,
    checkpointing_system,
    plot_evaluation,
    training_pipeline,
    metrics_system,
    Logger,
)

parser = argparse.ArgumentParser(description="Train")
parser.add_argument(
    "tfrecords",
    type=str,
    metavar="TFRECORDS_DIR",
    help="train using all tfrecords in TFRECORDS_DIR",
)
parser.add_argument(
    "output",
    type=str,
    metavar="OUR_DIR",
    help="save metrics and checkpoints in OUR_DIR",
)
parser.add_argument(
    "--run-id", type=str, required=True, metavar="RUN_ID", help="unique run identifier"
)
parser.add_argument(
    "--train-steps",
    type=int,
    required=True,
    metavar="STEPS",
    help="train for STEPS steps",
)
parser.add_argument(
    "--val-every",
    type=int,
    required=True,
    metavar="STEPS",
    help="run validation and save checkpoint every STEPS steps",
)
args = parser.parse_args()
dir_tfrecords = Path(args.tfrecords).expanduser().resolve().as_posix()
dir_output = Path(args.output).expanduser().resolve().joinpath(args.run_id).as_posix()

# PARAMETERS
run_id = args.run_id
val_number_of_images = 10
batch_size = 100
learning_rate = 0.001

# START
sess = tf.Session()
K.set_session(sess)

# Build the network and the various operations
col = Colorization(256)
opt_operations = training_pipeline(col, learning_rate, batch_size, dir_tfrecords)
evaluations_ops = evaluation_pipeline(col, val_number_of_images, dir_tfrecords)
summary_writer = metrics_system(sess, dir_output)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(dir_output)
logger = Logger(f"{dir_output}/output.txt")

with sess.as_default():
    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Restore
    if latest_checkpoint is not None:
        logger.write("Restoring from: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        logger.write(" done!")
    else:
        logger.write("No checkpoint found in: {}".format(checkpoint_paths))

    # Training loop
    for batch_idx in range(args.train_steps):
        logger.write(f"Batch: {batch_idx}/{args.train_steps}")
        res = sess.run(opt_operations)
        global_step = res["global_step"]
        logger.write(f'Cost: {res["cost"]} Global step: {global_step}')
        summary_writer.add_summary(res["summary"], global_step)

        if (batch_idx + 1) % args.val_every == 0:
            # Save the variables to disk
            save_path = saver.save(sess, checkpoint_paths, global_step)
            logger.write("Model saved in: %s" % save_path)

            # Evaluation step
            res = sess.run(evaluations_ops)
            summary_writer.add_summary(res["summary"], global_step)
            plot_evaluation(res, global_step, dir_output)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
