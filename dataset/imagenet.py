import argparse
import hashlib
import sys
import tarfile
import urllib.request
from itertools import islice
from os.path import join
from typing import Union, List

from .shared import *


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


def image_name(image_url: str) -> str:
    return hashlib.md5(image_url.encode('utf-8')).hexdigest() + '.jpg'


def get_image(image_url: str) -> Union[str, None]:
    image_path = join(dir_originals, image_name(image_url))
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
                split = line.split()
                if len(split) != 2:
                    continue
                yield split[1]
            except UnicodeDecodeError as ue:
                print('Unicode error: {}'.format(ue), file=sys.stderr)


def get_images(size=10, skip=0) -> List[str]:
    urls = islice(image_urls_generator(), skip, skip + size)
    valid_images = filter(lambda x: x is not None, (get_image(url) for url in urls))
    return list(valid_images)


if __name__ == '__main__':
    # Argparse setup
    parser = argparse.ArgumentParser(description='Download and process images from imagenet')
    parser.add_argument('-c', '--count', default=10, type=int,
                        help='get COUNT images (default: 10)')
    parser.add_argument('-f', '--source-file', default=links_source_url, type=str,
                        help='set the path for the links source file (default: {})'
                        .format(links_source_url), metavar='FILE', dest='source', )
    parser.add_argument('-o', '--output-folder', default=dir_originals, type=str,
                        help='use FOLDER to store the images (default: {})'
                        .format(dir_originals), metavar='FOLDER', dest='output', )

    args = parser.parse_args()
    dir_originals = args.output

    # Set up folders
    maybe_create_folder(dir_originals)

    # Download images
    for img_path in get_images(args.count):
        print(img_path)
