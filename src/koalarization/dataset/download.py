import argparse
import hashlib
import imghdr
import sys
import tarfile
import urllib.request
from itertools import islice
from os.path import join, isfile
from typing import Union, List

from koalarization.dataset.shared import DIR_ROOT, DIR_ORIGINALS, FILE_IMAGEURLS, maybe_create_folder


class ImagenetDownloader:
    """Class instance to download the images"""

    def __init__(self, links_source, dest_dir):
        """Constructor.

        Args:
            links_source (str): Link or path to file containing dataset URLs. Use local file to boost performance.
            dest_dir (str): Destination folder to save downloaded images.

        """
        print(links_source)
        # Destination folder
        maybe_create_folder(dest_dir)
        self.dest_dir = dest_dir
        # If the source is a link download it
        if links_source.startswith('http://'):
            print('Using urllib.request for the link archive is extremely',
                  'slow, it\'s better to download the tgz archive manualy',
                  'and pass its path to this constructor', file=sys.stderr)
            links_source, _ = urllib.request.urlretrieve(
                links_source,
                join(DIR_ROOT, 'imagenet_fall11_urls.txt')
            )

        # If the source is an archive extract it
        if links_source.endswith('.tgz'):
            with tarfile.open(links_source, 'r:gz') as tar:
                tar.extractall(path=DIR_ROOT)
                links_source = join(DIR_ROOT, 'fall11_urls.txt')

        # if not isfile(links_source):
        #     raise Exception('Links source not valid: {}'.format(links_source))

        self.links_source = links_source

    def download_images(self, size=10, skip=0):
        """Download images.

        Args:
            size (int, optional): Number of images to download. Defaults to 10.
            skip (int, optional): Number of images to skip at first. Defaults to 0.

        Returns:
            List[str]: List with image paths.

        """
        urls = islice(self._image_urls_generator(), skip, skip + size)
        downloaded_images = map(self._download_img, urls)
        valid_images = filter(lambda x: x is not None, downloaded_images)
        return list(valid_images)

    def _download_img(self, image_url: str):
        """Download single image.

        Args:
            image_url (str): Image url.

        Returns:
            Union[str, None]: Image path if image was succesfully downloaded. Otherwise, None.
        """
        image_name = self._encode_image_name(image_url)
        image_path = join(self.dest_dir, image_name)
        if not isfile(image_path):
            try:
                request = urllib.request.urlopen(image_url, timeout=2)
                image = request.read()
                if imghdr.what('', image) == 'jpeg':
                    with open(image_path, "wb") as f:
                        f.write(image)
            except Exception as e:
                print('Error downloading {}: {}'.format(image_url, e),
                      file=sys.stderr)
                return None
        return image_path

    def _image_urls_generator(self):
        """Generate image URL.

        Returns:
            Union[str, None]: List of image URLs.

        Yields:
            Iterator[Union[str, None]]: Iterator over image URLs.

        """
        with open(self.links_source) as sources:
            while True:
                try:
                    line = sources.readline()
                    split = line.split()
                    if len(split) == 2:
                        yield split[1]
                    elif line == '':
                        # End of file
                        return
                except UnicodeDecodeError as ue:
                    print('Unicode error: {}'.format(ue), file=sys.stderr)

    @staticmethod
    def _encode_image_name(image_url: str) -> str:
        """Image name encoding.

        Args:
            image_url (str): Image URL.

        Returns:
            str: Encoded image name.

        """
        encoded_name = '{}.jpeg'.format(hashlib.md5(image_url.encode('utf-8')).hexdigest())
        return encoded_name


def _parse_args():
    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Download and process images from imagenet'
    )
    parser.add_argument('-c', '--count',
                        default=10,
                        type=int,
                        help='get COUNT images (default: 10)')
    parser.add_argument('--skip',
                        default=0,
                        type=int,
                        metavar='N',
                        help='skip the first N images (default: 0)')
    parser.add_argument('-s', '--source',
                        default=FILE_IMAGEURLS,
                        type=str,
                        dest='source',
                        help='set source for the image links, can be the url, the archive or the file itself '
                             '(default: {})'
                        .format(FILE_IMAGEURLS))
    parser.add_argument('-o', '--output-folder',
                        default=DIR_ORIGINALS,
                        type=str,
                        metavar='FOLDER',
                        dest='output',
                        help='use FOLDER to store the images (default: {})'
                        .format(DIR_ORIGINALS))

    args = parser.parse_args()
    return args


# Run from the top folder as:
# python3 -m dataset.download <args>
if __name__ == '__main__':
    links_url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
    # links_url = http://www.image-net.org/image/tiny/tiny-imagenet-200.zip
    links_url = 'http://media.githubusercontent.com/media/akando42/1stPyTorch/master/fall11_urls.txt'

    args = _parse_args()
    ImagenetDownloader(
        links_source=args.source, 
        dest_dir=args.output
    ).download_images(
        size=args.count, 
        skip=args.skip
    )
