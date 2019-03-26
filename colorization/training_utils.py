import pickle
import time
from os.path import join

import matplotlib
import numpy as np
import cv2
from skimage import color
from PIL import Image, ImageChops

from dataset.shared import dir_tfrecord, dir_metrics, dir_checkpoints, dir_root, \
    maybe_create_folder
from dataset.tfrecords import LabImageRecordReader

# import datetime for clocking training speed per epoch
from datetime import datetime
prev_time = "00:00:00.000000"

matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (14.0, 4.0)
import matplotlib.pyplot as plt
import tensorflow as tf


labels_to_categories = pickle.load(
    open(join(dir_root, 'imagenet1000_clsid_to_human.pkl'), 'rb'))


def loss_with_metrics(img_ab_out, img_ab_true, name=''):
    # Loss is mean square error
    cost = tf.reduce_mean(
        tf.squared_difference(img_ab_out, img_ab_true), name="mse")
    # Metrics for tensorboard
    summary = tf.summary.scalar('cost ' + name, cost)
    return cost, summary


def training_pipeline(col, ref, learning_rate, batch_size):
    # Set up training (input queues, graph, optimizer)
    irr = LabImageRecordReader('lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(batch_size, shuffle=True)
    # read_batched_examples = irr.read_one()
    imgs_l = read_batched_examples['image_l']
    imgs_true_ab = read_batched_examples['image_ab']
    # Concatenate imgs_l and imgs_true_ab as imgs_lab to train on Refinement Network
    imgs_lab = tf.concat([imgs_l, imgs_true_ab], axis=3)
    imgs_emb = read_batched_examples['image_embedding']
    imgs_ab = col.build(imgs_l, imgs_emb)
    imgs_ref_ab = ref.build(imgs_lab)
    cost, summary = loss_with_metrics(imgs_ab, imgs_true_ab, 'training')
    cost_ref, summary_ref = loss_with_metrics(imgs_ref_ab, imgs_true_ab, 'training_ref')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        cost, global_step=global_step)
    optimizer_ref = tf.train.AdamOptimizer(learning_rate).minimize(
        cost_ref, global_step=global_step)
    return {
        'global_step': global_step,
        'optimizer': optimizer,
        'optimizer_ref': optimizer_ref,
        'cost': cost,
        'cost_ref': cost_ref,
        'summary': summary,
        'summary_ref': summary_ref,
    }#, irr, read_batched_examples


def evaluation_pipeline(col, ref, number_of_images):
    # Set up validation (input queues, graph)
    irr = LabImageRecordReader('val_lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(number_of_images, shuffle=False)
    imgs_l_val = read_batched_examples['image_l']
    imgs_true_ab_val = read_batched_examples['image_ab']
    imgs_emb_val = read_batched_examples['image_embedding']
    imgs_ab_val = col.build(imgs_l_val, imgs_emb_val)
    # Concatenate imgs_l_val and imgs_ab_val as imgs_lab_val to eval on Refinement Network
    imgs_lab_val = tf.concat([imgs_l_val, imgs_ab_val], axis=3)
    imgs_ref_ab_val = ref.build(imgs_lab_val)
    cost, summary = loss_with_metrics(imgs_ab_val, imgs_true_ab_val,
                                      'validation')
    cost_ref, summary_ref = loss_with_metrics(imgs_ref_ab_val, imgs_true_ab_val,
                                      'validation_ref')
    return {
        'imgs_l': imgs_l_val,
        'imgs_ab': imgs_ab_val,
        'imgs_true_ab': imgs_true_ab_val,
        'imgs_emb': imgs_emb_val,
        'imgs_lab': imgs_lab_val,
        'imgs_ref_ab': imgs_ref_ab_val,
        'cost': cost,
        'summary': summary,
        'cost_ref': cost_ref,
        'summary_ref': summary_ref,
    }


def print_log(content, run_id):
    with open('output_{}.txt'.format(run_id), mode='a') as f:
        f.write('[{}] {}\n'.format(time.strftime("%c"), content))


def print_term(content, run_id, cost=None):
    global prev_time
    curr_time = datetime.now().strftime("%H:%M:%S.%f")
    FMT = '%H:%M:%S.%f'
    time_diff = datetime.strptime(curr_time, FMT) - datetime.strptime(prev_time, FMT) if "Global step" in content else ""
    # print('{}[{}][{}] {}\n'.format(run_id, time.strftime("%c"), time_diff, content))
    print_log(content, run_id)
    # write on the output_train_time_per_batch_*.txt file the train_time_time_per_batch or time_diff 
    if time_diff:
        # tf.summary.scalar('time_diff', time_diff)
        with open('output_train_time_per_batch_{}.txt'.format(run_id), mode='a') as f:
            f.write('{}\n'.format(time_diff))
    # if cost:
    #     with open('output_cost_{}.txt'.format(run_id), mode='a') as f:
    #         f.write('{}\n'.format(cost))
    prev_time = curr_time


def metrics_system(run_id, sess):
    # Merge all the summaries and set up the writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(join(dir_metrics, run_id), sess.graph)
    return train_writer


def checkpointing_system(run_id):
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    checkpoint_paths = join(dir_checkpoints, run_id)
    latest_checkpoint = tf.train.latest_checkpoint(dir_checkpoints)
    return saver, checkpoint_paths, latest_checkpoint


def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rgMean, rgStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


def trim(im):
    im = im * 255
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def plot_evaluation(res, run_id, epoch, is_eval=False):
    maybe_create_folder(join(dir_root, 'images', run_id))
    for k in range(len(res['imgs_l'])):
        imgs_l = res['imgs_l'][k][:, :, 0]
        l_shape = imgs_l.shape
        img_gray = l_to_rgb(imgs_l)
        zeros = np.zeros(l_shape)
        img_ab = lab_to_rgb(zeros,
                                res['imgs_ab'][k])
        img_ref_ab = lab_to_rgb(zeros,
                                res['imgs_ref_ab'][k])
        img_output = lab_to_rgb(imgs_l,
                                res['imgs_ab'][k])
        img_ref_output = lab_to_rgb(imgs_l,
                                res['imgs_ref_ab'][k])
        
        # save simple single image output
        if is_eval:
            im = trim(img_ref_output)
            im.save(join(dir_root, 'images', run_id, '{}.png'.format(k)), "PNG")

        # display the colorfulness score on the image 
        C_output = image_colorfulness(img_output)
        C_ref_output = image_colorfulness(img_ref_output)
        img_true = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                              res['imgs_true_ab'][k])
        C_true = image_colorfulness(img_true)
        # display the cost function(MSE) output of the image
        cost = res['cost']

        plt.subplot(1, 6, 0)
        plt.imshow(img_gray)
        plt.title('Input (grayscale)')
        plt.axis('off')
        plt.subplot(1, 6, 1)
        plt.imshow(img_ab)
        plt.title('Colorization ab')
        plt.axis('off')
        plt.subplot(1, 6, 2)
        plt.imshow(img_ref_ab)
        plt.title('Refined ab')
        plt.axis('off')
        plt.subplot(1, 6, 3)
        plt.imshow(img_output)
        plt.title('Colorization output\n' + ("{:.4f}".format(C_output)))
        plt.axis('off')
        plt.subplot(1, 6, 4)
        plt.imshow(img_ref_output)
        plt.title('Refinement output\n' + ("{:.4f}".format(C_ref_output)))
        plt.axis('off')
        plt.subplot(1, 6, 5)
        plt.imshow(img_true)
        plt.title('Target (original)\n' + ("{:.4f}".format(C_true)))
        plt.axis('off')
        plt.suptitle('Cost(MSE): ' + str(cost), fontsize=7)

        plt.savefig(join(
            dir_root, 'images', run_id, '{}_{}.png'.format(epoch, k)))
        plt.clf()
        plt.close()

        # write on the output_colorfulness_*.txt file the colorfulness of the output image and the groundtruth image
        with open('output_colorfulness_{}.txt'.format(run_id), mode='a') as f:
            f.write('{},{}\n'.format(C_output, C_true))


def l_to_rgb(img_l):
    """
    Convert a numpy array (l channel) into an rgb image
    :param img_l:
    :return:
    """
    lab = np.squeeze(255 * (img_l + 1) / 2)
    return color.gray2rgb(lab) / 255


def lab_to_rgb(img_l, img_ab):
    """
    Convert a pair of numpy arrays (l channel and ab channels) into an rgb image
    :param img_l:
    :return:
    """
    lab = np.empty([*img_l.shape[0:2], 3])
    lab[:, :, 0] = np.squeeze(((img_l + 1) * 50))
    lab[:, :, 1:] = img_ab * 127
    return color.lab2rgb(lab)
