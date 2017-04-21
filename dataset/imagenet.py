import os
import sys
import tarfile
import urllib.request

import hashlib

from itertools import islice

# NOTE urllib.request is extremely slow, it's better to download
# the tgz archive separately extract fall11_urls.txt and put it
# into the current folder

class Batch:
    def __init__(self, images):
        self.size = len(images)
        self.images = images

    def __str__(self):
        return 'Batch (total images {})\n'.format(self.size)


class ImageNet:
    links_source_url = "http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz"
    links_file = 'fall11_urls.txt'
    images_folder = 'imagenet_images'

    @staticmethod
    def __check_links_file_exists():
        if not os.path.isfile(ImageNet.links_file):
            file, _ = urllib.request.urlretrieve(
                ImageNet.links_source_url,
                'temp.tgz')
            file = 'temp.tgz'
            with tarfile.open(file, 'r:gz') as tar:
                tar.extractall()
            os.remove(file)
            with open('.gitignore', 'a+') as out:
                out.write(ImageNet.links_file + '\n')

    @staticmethod
    def __check_imagenet_folder_exists():
        if not os.path.isdir(ImageNet.images_folder):
            os.mkdir(ImageNet.images_folder)
            with open('.gitignore', 'a+') as out:
                out.write(ImageNet.images_folder + '\n')

    @staticmethod
    def __image_path(image_url: str):
        return '{}/{}.jpeg'.format(
            ImageNet.images_folder,
            hashlib.md5(image_url.encode('utf-8')).hexdigest())

    @staticmethod
    def __get_image(image_url: str) -> str:
        print(image_url)
        image_path = ImageNet.__image_path(image_url)
        if not os.path.isfile(image_path):
            try:
                ImageNet.__check_imagenet_folder_exists()
                urllib.request.urlretrieve(
                    image_url,
                    image_path)
            except Exception as e:
                print('Error downloading {}: {}'.format(image_url, e), file=sys.stderr)
                return None
        return image_path

    @staticmethod
    def image_urls_generator(limit=None):
        return islice(ImageNet.all_image_urls_generator(), limit)

    @staticmethod
    def all_image_urls_generator():
        ImageNet.__check_links_file_exists()
        with open(ImageNet.links_file) as links_file:
            while True:
                try:
                    line = links_file.readline()
                    if line == '':
                        return
                    yield line.split()[1]
                except UnicodeDecodeError as ue:
                    print('Unicode error after line {}: {}'.format(line, ue), file=sys.stderr)

    @staticmethod
    def get_images(size=None) -> Batch:
        return Batch(
            images=list(filter(lambda x: x is not None,
                               (ImageNet.__get_image(url) for url in ImageNet.image_urls_generator(size))))
        )


if __name__ == '__main__':
    im = ImageNet.get_images(15)

    for i in im.images:
        print(i)
