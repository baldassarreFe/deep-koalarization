import argparse
from pathlib import Path

import tensorflow as tf
from keras import backend as K

from .network_definition import Colorization
from .training_utils import (
    evaluation_pipeline,
    checkpointing_system,
    plot_evaluation,
    metrics_system,
)

parser = argparse.ArgumentParser(description="Eval")
parser.add_argument(
    "tfrecords",
    type=str,
    metavar="TFRECORDS_DIR",
    help="evaluate using all tfrecords in TFRECORDS_DIR",
)
parser.add_argument(
    "output",
    type=str,
    metavar="OUR_DIR",
    help="use OUR_DIR to load checkpoints and write images",
)
parser.add_argument(
    "--run-id",
    required=True,
    type=str,
    metavar="RUN_ID",
    help="load checkpoint from the run RUN_ID",
)
args = parser.parse_args()
dir_tfrecords = Path(args.tfrecords).expanduser().resolve().as_posix()
dir_output = Path(args.output).expanduser().resolve().joinpath(args.run_id).as_posix()

# PARAMETERS
run_id = args.run_id
val_number_of_images = 100

# START
sess = tf.Session()
K.set_session(sess)

# Build the network and the various operations
col = Colorization(256)
evaluations_ops = evaluation_pipeline(col, val_number_of_images, dir_tfrecords)
summary_writer = metrics_system(sess, dir_output)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(dir_output)

with sess.as_default():
    # Initialize
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Restore
    if latest_checkpoint is not None:
        print(f"Restoring from: {latest_checkpoint}")
        saver.restore(sess, latest_checkpoint)
    else:
        print(f"No checkpoint found in: {checkpoint_paths}")
        exit(1)

    res = sess.run(evaluations_ops)
    print("Cost: {}".format(res["cost"]))
    plot_evaluation(res, "eval", dir_output)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
