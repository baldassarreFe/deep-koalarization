import pickle
import time
from os.path import join

import matplotlib
import numpy as np
import cv2
from skimage import color

from dataset.shared import dir_tfrecord, dir_metrics, dir_checkpoints, dir_root, \
    maybe_create_folder
from dataset.tfrecords import LabImageRecordReader

# import datetime for clocking training speed per epoch
from datetime import datetime
prev_time = "00:00:00.000000"

matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (10.0, 4.0)
import matplotlib.pyplot as plt
import tensorflow as tf


labels_to_categories = pickle.load(
    open(join(dir_root, 'imagenet1000_clsid_to_human.pkl'), 'rb'))


def loss_with_metrics(img_ab_out, img_ab_true, name=''):
    # Loss is mean square erros
    cost = tf.reduce_mean(
        tf.squared_difference(img_ab_out, img_ab_true), name="mse")
    # Metrics for tensorboard
    summary = tf.summary.scalar('cost ' + name, cost)
    return cost, summary


def training_pipeline(col, learning_rate, batch_size):
    # Set up training (input queues, graph, optimizer)
    irr = LabImageRecordReader('lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(batch_size, shuffle=True)
    # read_batched_examples = irr.read_one()
    imgs_l = read_batched_examples['image_l']
    imgs_true_ab = read_batched_examples['image_ab']
    imgs_emb = read_batched_examples['image_embedding']
    imgs_ab = col.build(imgs_l, imgs_emb)
    cost, summary = loss_with_metrics(imgs_ab, imgs_true_ab, 'training')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        cost, global_step=global_step)
    return {
        'global_step': global_step,
        'optimizer': optimizer,
        'cost': cost,
        'summary': summary
    }#, irr, read_batched_examples


def evaluation_pipeline(col, number_of_images):
    # Set up validation (input queues, graph)
    irr = LabImageRecordReader('val_lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(number_of_images, shuffle=False)
    imgs_l_val = read_batched_examples['image_l']
    imgs_true_ab_val = read_batched_examples['image_ab']
    imgs_emb_val = read_batched_examples['image_embedding']
    imgs_ab_val = col.build(imgs_l_val, imgs_emb_val)
    cost, summary = loss_with_metrics(imgs_ab_val, imgs_true_ab_val,
                                      'validation')
    return {
        'imgs_l': imgs_l_val,
        'imgs_ab': imgs_ab_val,
        'imgs_true_ab': imgs_true_ab_val,
        'imgs_emb': imgs_emb_val,
        'cost': cost,
        'summary': summary
    }


def print_log(content, run_id):
    with open('output_{}.txt'.format(run_id), mode='a') as f:
        f.write('[{}] {}\n'.format(time.strftime("%c"), content))

def print_term(content, run_id, cost=None):
    global prev_time
    curr_time = datetime.now().strftime("%H:%M:%S.%f")
    FMT = '%H:%M:%S.%f'
    time_diff = datetime.strptime(curr_time, FMT) - datetime.strptime(prev_time, FMT) if "Global step" in content else ""
    print('{}[{}][{}] {}\n'.format(run_id, time.strftime("%c"), time_diff, content))
    # write on the output_train_time_per_batch_*.txt file the train_time_time_per_batch or time_diff 
    with open('output_train_time_per_batch_{}.txt'.format(run_id), mode='a') as f:
        f.write('{}\n'.format(time_diff))
    if cost:
        with open('output_cost_{}.txt'.format(run_id), mode='a') as f:
            f.write('{}\n'.format(cost))
    prev_time = curr_time

def metrics_system(run_id, sess):
    # Merge all the summaries and set up the writers
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


def crop_image(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    print(mask)
    print(type(mask))
    import sys
    sys.exit()
    return img[np.ix_(mask.any(1),mask.any(0))]


from PIL import Image, ImageChops
from matplotlib import cm


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def trim(im):
    '''
    img = im
    
    mean = 0
    # var = 0.1
    # sigma = var**0.5
    gauss = np.random.normal(mean, 1, img.shape)

    # normalize image to range [0,255]
    noisy = img + gauss
    minv = np.amin(noisy)
    maxv = np.amax(noisy)
    noisy = (255 * (noisy - minv) / (maxv - minv)).astype(np.uint8)

    #im = Image.fromarray(img)
    '''
    size = im.shape
    size = (size[0], size[1])
    #im = np.interp(im, (im.min(), im.max()), (0, 255))    
    im = im * 255
    #print(im)
    #import sys
    #sys.exit()
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    #im = array2PIL(im, size)#Image.fromarray(np.rollaxis(img, 0,3))
    # im = Image.fromarray(np.uint8(cm.gist_earth(im, bytes=True)))


    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff)#, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def crop(data_to_crop):
    # img = data_to_crop
    # for x in range(data_to_crop.shape[0]):
    #     for y in range(data_to_crop.shape[1]):
    #         px = img[]
    cropped_data = []

    for z in range(data_to_crop.shape[0]):
        img = data_to_crop#[z]
        #img = img.reshape(data_to_crop.shape[0], data_to_crop.shape[1])
        rx = -1
        upy = -1
        lx = -1
        by = -1
        for x in range(data_to_crop.shape[0]):
            for y in range(data_to_crop.shape[1]):
                px = img[x, y]
                if any(px > 0):
                    if rx == -1 or x > rx:
                        rx = x
                    if lx == -1 or x < lx:
                        lx = x
                    if upy == -1 or y > upy:
                        upy = y
                    if by == -1 or y < by:
                        by = y
        img = img[lx:rx, by:upy]
        cropped_data.append(img)
    return cropped_data

def crap(img):
    pass 

def plot_evaluation(res, run_id, epoch, is_eval=False):
    maybe_create_folder(join(dir_root, 'images', run_id))
    for k in range(len(res['imgs_l'])):
        img_gray = l_to_rgb(res['imgs_l'][k][:, :, 0])
        img_output = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                                res['imgs_ab'][k])
        
        # save simple single image output
        if is_eval:
            plt.axis('off')
            #plt.tight_layout()
            
            '''
            # Mask of non-black pixels (assuming image has a single channel).
            mask = img_output < 1

            # Coordinates of non-black pixels.
            coords = np.argwhere(mask)
            print(type(coords))
            print(coords.shape)

            # Bounding box of non-black pixels.
            x0 = coords.min(axis=0)
            y0 = coords.max(axis=0) + 1   # slices are exclusive at the top
            x1 = coords.min(axis=1)
            y1 = coords.max(axis=1) + 1

            print('x0: ' + str(x0) + ', y0: ' + str(y0) + ', x1: ' + str(x1) + ', y1: ' + str(y1))
            #import sys
            #sys.exit()

            # Get the contents of the bounding box.
            img_output = img_output[x0:x1, y0:y1]
            '''
            #print(img_output)
            #img_output = crop_image(img_output)
            #print(img_output[::2])
            #import sys
            #sys.exit()
            #img = img_output
            #img_output[::][0]
            #mask = img_output>0 
            #print(mask.any(0))
            #print(mask)
            #print(img_output[mask])
            #import sys
            #sys.exit()
            #img_output = img_output[np.ix_(mask.any(1), mask.any(0))]#img_output[49:150, 49:150]
            
            '''
            >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            '''
            
            '''image_data = img_output
            image_data_bw = image_data.max(axis=2)

            non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
            non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
            cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

            image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
            img_trimmed = image_data_new
            print(img_trimmed.shape)
            '''


            # img_trimmed = crop(img_output)
            #img_trimmed = trim(img_output)
            #img_trimmed = PIL2array(img_trimmed)
            img_trimmed = img_output
            size = img_trimmed.shape
            size = (size[0], size[1])
            #print(size)
            #import sys
            #sys.exit()
            #img_trimmed = img_trimmed[1:size[0] - 50, 50:size[1] - 1, :]

            #plt.imshow(img_trimmed)
            #plt.savefig(join(
            #    dir_root, 'images', run_id, '{}.png'.format(k)))#bbox_inches='tight'), pad_inches=0)
            #im = Image.fromarray(np.uint8(img_trimmed))
            #im = array2PIL(img_trimmed, size)
            im = img_trimmed * 255
            im = Image.fromarray(im.astype('uint8'), 'RGB')
            im.save(join(dir_root, 'images', run_id, '{}.png'.format(k)), "PNG")
            '''
            >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            '''



        C_output = image_colorfulness(img_output)
        # display the colorfulness score on the image
        cv2.putText(img_output, "{:.4f}".format(C_output), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        img_true = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                              res['imgs_true_ab'][k])
        C_true = image_colorfulness(img_true)
        cv2.putText(img_true, "{:.4f}".format(C_true), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        # display the cost function(MSE) output of the image
        cost = res['cost']
        '''
        top_5 = np.argsort(res['imgs_emb'][k])[-5:]
        try:
            top_5 = ' / '.join(labels_to_categories[i] for i in top_5)
        except:
            ptop_5 = str(top_5)
        '''

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
        plt.suptitle('Cost(MSE): ' + str(cost), fontsize=7)
        # plt.suptitle(top_5, fontsize=7)

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
