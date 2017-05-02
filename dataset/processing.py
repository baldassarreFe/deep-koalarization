import argparse
import os
import sys

from PIL import Image
from resizeimage import resizeimage
from shared import *


def resize(source_path, output_path, size=(299, 299)):
    """
    Resizes the image using padding
    :param source_path:
    :param output_path:
    :param size:
    :return:
    """
    img = Image.open(source_path)
    cover = resizeimage.resize_contain(img, size)
    cover.save(output_path, 'JPEG')


def filter(source_path, output_path, size=(299, 299)):
    pass


def processing():
    file_list = (f for f in os.listdir(dir_originals)
                 if os.path.isfile(os.path.join(dir_originals, f)))
    for f in file_list:
        resize(os.path.join(dir_originals, f), os.path.join(dir_resized, f))
        filter(os.path.join(dir_originals, f), os.path.join(dir_filtered, f))


if __name__ == '__main__':
    # Argparse setup
    parser = argparse.ArgumentParser(description='Download and process images from imagenet')
    parser.add_argument('-i', '--source-folder', default=dir_originals, type=str,
                        help='use FOLDER as source of the images (default: {})'
                        .format(dir_originals), metavar='FOLDER', dest='source', )
    parser.add_argument('-r', '--resized-folder', default=dir_resized, type=str,
                        help='use FOLDER for the resized images (default: {})'
                        .format(dir_resized), metavar='FOLDER', dest='resized', )
    parser.add_argument('-f', '--filtered-folder', default=dir_filtered, type=str,
                        help='use FOLDER for the filtered images (default: {})'
                        .format(dir_filtered), metavar='FOLDER', dest='filtered', )

    args = parser.parse_args()
    dir_originals = args.source
    dir_resized = args.resized
    dir_filtered = args.filtered

    # Set up folders
    if not os.path.isdir(dir_originals):
        print('Input folder does not exists: {}'.format(dir_originals), file=sys.stderr)
        exit(-1)
    maybe_create_folder(dir_resized)
    maybe_create_folder(dir_filtered)

    processing()
