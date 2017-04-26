import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
from itertools import islice
from PIL import Image
#TO INSTALL: pip install python-resize-image
from resizeimage import resizeimage

from typing import Union, List


# TODO We want this batch to be a tensorflow-compatible batch
class Batch:
    def __init__(self, images):
        self.size = len(images)
        self.images = images

    def __str__(self):
        return 'Batch (total images {})\n'.format(self.size)


links_source_url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
compressed_links_file = 'imagenet_fall11_urls.tgz'
links_file = 'fall11_urls.txt'
dir_root = 'imagenet'
dir_originals = 'original/'
dir_resized = 'resized/'
dir_filtered = 'filtered/'


def check_folders_exist(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


# NOTE:
# Using urllib.request for the link archive is extremely slow,
# it's better to download the tgz archive separately put it
# into the current folder
def check_links_file_exists():
    if not os.path.isfile(links_file):
        if not os.path.isfile(compressed_links_file):
            urllib.request.urlretrieve(links_source_url, compressed_links_file)
        with tarfile.open(compressed_links_file, 'r:gz') as tar:
            tar.extractall()


def image_name(image_url: str, suffix='') -> str:
    return '{}{}.jpeg'.format(
        hashlib.md5(image_url.encode('utf-8')).hexdigest(),
        suffix)


def get_image(image_url: str) -> Union[str, None]:
    image_path = dir_root + '/' + dir_originals + image_name(image_url)
    if not os.path.isfile(image_path):
        try:
            urllib.request.urlretrieve(image_url, image_path)
        except Exception as e:
            print('Error downloading {}: {}'.format(image_url, e), file=sys.stderr)
            return None
    return image_path


def image_urls_generator():
    check_links_file_exists()
    with open(links_file) as sources:
        while True:
            try:
                line = sources.readline()
                if line == '':
                    return
                yield line.split()[1]
            except UnicodeDecodeError as ue:
                print('Unicode error: {}'.format(ue), file=sys.stderr)


def get_images(size=10, skip=0) -> List[str]:
    urls = islice(image_urls_generator(), skip, skip + size)
    valid_images = filter(lambda x: x is not None, (get_image(url) for url in urls))
    return list(valid_images)

def Resize_Img ( filename , save_path , size = [299, 299] ):
	img = Image.open( filename )
	# Resize using padding
	cover = resizeimage.resize_contain(img, size)
	# Make dir if it does not exist
	try:
		os.stat(save_path)
	except:
		os.mkdir(save_path)  	
	# Safe resized image
	cover.save( save_path + filename + '-resized.jpeg', img.format)


if __name__ == '__main__':
    # Argparse setup
    parser = argparse.ArgumentParser(description='Download and process images from imagenet')
    parser.add_argument('-c', '--count', default=10, type=int,
                        help='get COUNT images (default: 10)')
    parser.add_argument('-f', '--source-file', default=None, type=str, metavar='FILE', dest='source',
                        help='set the path for the links source file')
    parser.add_argument('-o', '--image-folder', default=None, type=str, metavar='FOLDER', dest='root',
                        help='use FOLDER as root of the images folders (default: imagenet)')

    # Using the arguments
    args = parser.parse_args()
    if args.source is not None:
        links_file = args.source
    if args.root is not None:
        dir_root = args.root

    # Set up folders
    check_folders_exist(dir_root)
    check_folders_exist(dir_root + '/' + dir_originals)
    check_folders_exist(dir_root + '/' + dir_resized)
    check_folders_exist(dir_root + '/' + dir_filtered)

    for img_path in get_images(args.count):
        print(img_path)
