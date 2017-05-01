import argparse
import os
import sys
import unittest
import cv2
import Filters
import ImageOperator
import shutil 

from filter_functions.filters import Filters
from filter_functions.image_operation import ImageOperator
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
    cover.save(output_path, img.format)

def filters(source_path, output_path, namesCV = ('nashville', 'gotham', 'claredon'), namesIM = ('lomo', 'kelvin', 'nash2', 'toaster')):
    	out = []		
	IO = ImageOperator()
	filters = Filters()
	IO.image(source_path)
	inImage = IO.im
	
	# OPEN CV FILTERS
	out.append( filters.nashville(inImage))
	out.append( filters.gotham(inImage))
	out.append( filters.claredon(inImage))
	for i in range(len(namesCV)):
		cv2.imwrite(os.path.join('Transformed', namesCV[i] + '_' + source_path ),out[i])

	# IM filters
	filenameDumb = os.path.join('Transformed', namesIM[0] + '_' + source_path)
	shutil.copyfile(filename, filenameDumb)
	filters.lomo(filenameDumb)

	filenameDumb = os.path.join('Transformed', namesIM[1] + '_' + source_path )
	shutil.copyfile(source_path, filenameDumb)
	filters.kelvin(filenameDumb)

	filenameDumb = os.path.join('Transformed', namesIM[2] + '_' + source_path )
	shutil.copyfile(source_path, filenameDumb)
	filters.nash2(filenameDumb)

	filenameDumb = os.path.join('Transformed', namesIM[3] + '_' + source_path)
	shutil.copyfile(source_path, filenameDumb)
	filters.toaster(filenameDumb)

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
    resize(dir_originals, dir_resized)
    filters(dir_resized, dir_filtered)
