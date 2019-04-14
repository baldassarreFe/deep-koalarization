import pickle
import time
from os.path import join

import matplotlib
import numpy as np
import cv2
from skimage import color
from PIL import Image, ImageChops

import tensorboard
from tensorboard import summary as summary_lib

from dataset.shared import dir_tfrecord, dir_metrics, dir_checkpoints, dir_root, \
    maybe_create_folder
from dataset.tfrecords import LabImageRecordReader

# import datetime for clocking training speed per epoch
from datetime import datetime
prev_time = "00:00:00.000000"

matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (14.0, 8.0)
import matplotlib.pyplot as plt
import tensorflow as tf


labels_to_categories = pickle.load(
    open(join(dir_root, 'imagenet1000_clsid_to_human.pkl'), 'rb'))


def loss_with_metrics(img_ab_out, img_ab_true, name=''):
    # Loss is mean square error
    cost = tf.reduce_mean(
        tf.squared_difference(img_ab_out, img_ab_true), name="mse")
    # Metrics for tensorboard
    summary = summary_lib.scalar(name, cost)
    return cost, summary


def training_pipeline(col, lowres_col, ref, learning_rate, batch_size):
    # Set up training (input queues, graph, optimizer)
    irr = LabImageRecordReader('lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(batch_size, shuffle=True)
    # read_batched_examples = irr.read_one()
    imgs_l = read_batched_examples['image_l']
    imgs_true_ab = read_batched_examples['image_ab']
    imgs_emb = read_batched_examples['image_embedding']
    imgs_ab = col.build(imgs_l, imgs_emb)
    imgs_lowres_ab = lowres_col.build(imgs_l)#, imgs_emb)
    #shape = imgs_ab.shape
    #imgs_lowres_ab = tf.image.resize_images(imgs_lowres_ab, (shape[1], shape[2]))
    # Concatenate imgs_l, imgs_lowres_ab and imgs_true_ab as imgs_lab to train on Refinement Network
    imgs_lab = tf.concat([imgs_l, imgs_lowres_ab, imgs_ab], axis=3)
    imgs_ref_ab = ref.build(imgs_lab)
    cost, summary = loss_with_metrics(imgs_ab, imgs_true_ab, 'training_col')
    cost_lowres, summary_lowres = loss_with_metrics(imgs_lowres_ab, imgs_true_ab, 'training_lowres')
    cost_ref, summary_ref = loss_with_metrics(imgs_ref_ab, imgs_true_ab, 'training_ref')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        cost, global_step=global_step)
    optimizer_lowres = tf.train.AdamOptimizer(learning_rate).minimize(
        cost_lowres, global_step=global_step)
    optimizer_ref = tf.train.AdamOptimizer(learning_rate).minimize(
        cost_ref, global_step=global_step)
    return {
        'global_step': global_step,
        'optimizer': optimizer,
        'optimizer_lowres': optimizer_lowres,
        'optimizer_ref': optimizer_ref,
        'cost': cost,
        'cost_lowres': cost_lowres,
        'cost_ref': cost_ref,
        'summary': summary,
        'summary_lowres': summary_lowres,
        'summary_ref': summary_ref,
    }#, irr, read_batched_examples


def evaluation_pipeline(col, lowres_col, ref, number_of_images):
    # Set up validation (input queues, graph)
    irr = LabImageRecordReader('val_lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(number_of_images, shuffle=False)
    imgs_l_val = read_batched_examples['image_l']
    imgs_true_ab_val = read_batched_examples['image_ab']
    imgs_emb_val = read_batched_examples['image_embedding']
    imgs_ab_val = col.build(imgs_l_val, imgs_emb_val)
    imgs_lowres_ab_val = lowres_col.build(imgs_l_val)#, imgs_emb_val)
    #shape = imgs_ab_val.shape
    #imgs_lowres_ab_val = tf.image.resize_images(imgs_lowres_ab_val, (shape[1], shape[2]))
    # Concatenate imgs_l_val, imgs_lowres_ab_val and imgs_ab_val as imgs_lab_val to eval on Refinement Network
    imgs_lab_val = tf.concat([imgs_l_val, imgs_lowres_ab_val, imgs_ab_val], axis=3)
    imgs_ref_ab_val = ref.build(imgs_lab_val)
    cost, summary = loss_with_metrics(imgs_ab_val, imgs_true_ab_val,
                                      'validation_col')
    cost_lowres, summary_lowres = loss_with_metrics(imgs_lowres_ab_val, imgs_true_ab_val,
                                      'validation_lowres')
    cost_ref, summary_ref = loss_with_metrics(imgs_ref_ab_val, imgs_true_ab_val,
                                      'validation_ref')
    return {
        'imgs_l': imgs_l_val,
        'imgs_ab': imgs_ab_val,
        'imgs_lowres_ab': imgs_lowres_ab_val,
        'imgs_true_ab': imgs_true_ab_val,
        'imgs_emb': imgs_emb_val,
        'imgs_lab': imgs_lab_val,
        'imgs_ref_ab': imgs_ref_ab_val,
        'cost': cost,
        'cost_lowres': cost_lowres,
        'cost_ref': cost_ref,
        'summary': summary,
        'summary_lowres': summary_lowres,
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
    #merged = tf.summary.merge_all()
    train_col_writer = tf.summary.FileWriter(join(dir_metrics, run_id, 'train/col'), sess.graph)
    train_lowres_writer = tf.summary.FileWriter(join(dir_metrics, run_id, 'train/lowres'), sess.graph)
    train_ref_writer = tf.summary.FileWriter(join(dir_metrics, run_id, 'train/ref'), sess.graph)
    val_col_writer = tf.summary.FileWriter(join(dir_metrics, run_id, 'val/col'), sess.graph)
    val_lowres_writer = tf.summary.FileWriter(join(dir_metrics, run_id, 'val/lowres'), sess.graph)
    val_ref_writer = tf.summary.FileWriter(join(dir_metrics, run_id, 'val/ref'), sess.graph)
    return train_col_writer, train_lowres_writer, train_ref_writer, val_col_writer, val_lowres_writer, val_ref_writer


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
        img_gray = l_to_rgb(imgs_l)
        zeros = np.zeros(res['imgs_ref_ab'][k][:, :, 0].shape)
        img_ab = lab_to_rgb(zeros,
                                res['imgs_ab'][k])
        img_lowres_ab = lab_to_rgb(zeros,
                                res['imgs_lowres_ab'][k])
        img_ref_ab = lab_to_rgb(zeros,
                                res['imgs_ref_ab'][k])
        img_output = lab_to_rgb(imgs_l,
                                res['imgs_ab'][k])
        img_lowres_output = lab_to_rgb(imgs_l,
                                res['imgs_lowres_ab'][k])
        img_ref_output = lab_to_rgb(imgs_l,
                                res['imgs_ref_ab'][k])
        img_true = lab_to_rgb(imgs_l,
                              res['imgs_true_ab'][k])
        
        # save simple single image output
        if is_eval:
            im = trim(img_ref_output)
            im.save(join(dir_root, 'images', run_id, '{}.png'.format(k)), "PNG")

        # display the colorfulness score on the image 
        C_output = image_colorfulness(img_output)
        C_lowres_output = image_colorfulness(img_lowres_output)
        C_ref_output = image_colorfulness(img_ref_output)
        C_true = image_colorfulness(img_true)
        # display the cost function(MSE) output of the image
        cost = res['cost']

        fig, axes = plt.subplots(2, 4)
        # Colorization ab
        axes[0,0].imshow(img_ab)
        axes[0,0].set_title('Colorization ab')
        axes[0,0].axis('off')
        # Low res ab
        axes[0,1].imshow(img_lowres_ab)
        axes[0,1].set_title('Low resolution ab')
        axes[0,1].axis('off')
        # Refined ab
        axes[0,2].imshow(img_ref_ab)
        axes[0,2].set_title('Refined ab')
        axes[0,2].axis('off')
        # Input (grayscale)
        axes[0,3].imshow(img_gray)
        axes[0,3].set_title('Input (grayscale)')
        axes[0,3].axis('off')
        # Colorization output
        axes[1,0].imshow(img_output)
        axes[1,0].set_title('Colorization output\n' + ("{:.4f}".format(C_output)))
        axes[1,0].axis('off')
        # Low Resolution output
        axes[1,1].imshow(img_lowres_output)
        axes[1,1].set_title('Low Resolution output\n' + ("{:.4f}".format(C_lowres_output)))
        axes[1,1].axis('off')
        # Refinement output
        axes[1,2].imshow(img_ref_output)
        axes[1,2].set_title('Refinement output\n' + ("{:.4f}".format(C_ref_output)))
        axes[1,2].axis('off')
        # Target (original)
        axes[1,3].imshow(img_true)
        axes[1,3].set_title('Target (original)\n' + ("{:.4f}".format(C_true)))
        axes[1,3].axis('off')
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
