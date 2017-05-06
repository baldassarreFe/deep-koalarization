import hashlib
import imghdr
import sys
import tarfile
import urllib.request
from itertools import islice
from os.path import join, isfile
from typing import Union, List

from dataset.shared import dir_root, maybe_create_folder


class ImagenetDownloader:
    def __init__(self, links_source: str, dest_dir: str):

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
                join(dir_root, 'imagenet_fall11_urls.tgz'))

        # If the source is an archive extract it
        if links_source.endswith('.tgz'):
            with tarfile.open(links_source, 'r:gz') as tar:
                tar.extractall(path=dir_root)
                links_source = join(dir_root, 'fall11_urls.txt')

        if not isfile(links_source):
            raise Exception('Links source not valid: {}'.format(links_source))

        self.links_source = links_source

    def download_images(self, size: int = 10, skip: int = 0) -> List[str]:
        urls = islice(self._image_urls_generator(), skip, skip + size)
        downloaded_images = map(self._download_img, urls)
        valid_images = filter(lambda x: x is not None, downloaded_images)
        return list(valid_images)

    def _download_img(self, image_url: str) -> Union[str, None]:
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

    def _image_urls_generator(self) -> Union[str, None]:
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
        return hashlib.md5(image_url.encode('utf-8')).hexdigest() + '.jpeg'


# Run from the top folder as:
# python3 -m dataset.download <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_originals

    links_url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Download and process images from imagenet')
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
                        default=links_url,
                        type=str,
                        dest='source',
                        help='set the source for the image links, can be the url, the archive or the file itself (default: {})'
                        .format(links_url))
    parser.add_argument('-o', '--output-folder',
                        default=dir_originals,
                        type=str,
                        metavar='FOLDER',
                        dest='output',
                        help='use FOLDER to store the images (default: {})'
                        .format(dir_originals))

    args = parser.parse_args()
    ImagenetDownloader(args.source, args.output) \
        .download_images(args.count, args.skip)
