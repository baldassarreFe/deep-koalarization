import hashlib
import os
import sys
import tarfile
import urllib.request
from itertools import islice

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
dir_originals = 'imagenet_original/'
dir_resized = 'imagenet_resized/'
dir_filtered = 'imagenet_filtered/'


def check_folders_exist():
    if not os.path.isdir(dir_originals):
        os.mkdir(dir_originals)
    if not os.path.isdir(dir_resized):
        os.mkdir(dir_resized)
    if not os.path.isdir(dir_filtered):
        os.mkdir(dir_filtered)


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
    image_path = dir_originals + image_name(image_url)
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


if __name__ == '__main__':
    check_folders_exist()
    for img_path in get_images(10):
        print(img_path)