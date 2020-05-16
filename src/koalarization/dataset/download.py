"""Getting the images from Imagenet

To download ImageNet dataset, we provide a script which requires an input `txt` file containing the URLs to the images.

> Note: There used to be a file containing the image URLs for ImageNet 2011 available without registration on the
> official website. Since the link appears to be down, you may want to use a non-official file (see DATASET.md).

```
python -m koalarization.dataset.download urls.txt path/to/dest
```

Use `-h` to see the available options
"""


import argparse
import hashlib
import imghdr
import sys
import tarfile
import urllib.request
from itertools import islice
from os.path import join, isfile
from typing import List

from .shared import maybe_create_folder


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
        if links_source.startswith("http://"):
            print(
                "Using urllib.request for the link archive is extremely",
                "slow, it is better to download the tgz archive manually",
                "and pass its path to this constructor",
                file=sys.stderr,
            )
            links_source, _ = urllib.request.urlretrieve(
                links_source,
                'imagenet_fall11_urls.txt'
            )

        # If the source is an archive extract it
        if links_source.endswith('.tgz'):
            with tarfile.open(links_source, 'r:gz') as tar:
                tar.extractall(path='.')
                links_source = 'imagenet_fall11_urls.txt'

        # if not isfile(links_source):
        #     raise Exception('Links source not valid: {}'.format(links_source))

        self.links_source = links_source

    def download_images(self, size=None, skip=0):
        """Download images.

        Args:
            size (int, optional): Number of images to download. Defaults to all images.
            skip (int, optional): Number of images to skip at first. Defaults to 0.

        Returns:
            List[str]: List with image paths.

        """
        urls = self._image_urls_generator()
        urls = islice(urls, skip, None if size is None else skip+size)
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
                # TODO use request.get with accept jpg?
                request = urllib.request.urlopen(image_url, timeout=5)
                image = request.read()
                if imghdr.what("", image) == "jpeg":
                    with open(image_path, "wb") as f:
                        f.write(image)
            except Exception as e:
                print("Error downloading {}: {}".format(image_url, e), file=sys.stderr)
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
                    if line.startswith('#') or line == '\n':
                        # Comments or empty lines
                        continue
                    if line == '':
                        # End of file
                        return
                    url = line.rsplit(maxsplit=1)[-1]
                    if url.startswith('http'):
                        yield url
                except UnicodeDecodeError as ue:
                    print("Unicode error: {}".format(ue), file=sys.stderr)

    @staticmethod
    def _encode_image_name(image_url: str) -> str:
        """Image name encoding.

        Args:
            image_url (str): Image URL.

        Returns:
            str: Encoded image name.

        """

        hash = hashlib.md5(image_url.encode('utf-8')).hexdigest()
        encoded_name = f'{hash}.jpeg'
        return encoded_name


def _parse_args():
    """Get args.

    Returns:
        Namespace: Contains args

    """
    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Download and process images from a file of URLs.'
    )
    parser.add_argument(
        '-c', '--count',
        default=None,
        type=int,
        help='download only COUNT images (default all)'
    )
    parser.add_argument(
        '-s', '--skip',
        default=0,
        type=int,
        metavar='N',
        help='skip the first N images (default 0)'
    )
    parser.add_argument(
        'source',
        type=str,
        metavar='SOURCE',
        help='set source for the image links, can be the url, the archive or the file itself'
    )
    parser.add_argument(
        'output',
        default='.',
        type=str,
        metavar='OUT_DIR',
        help='save downloaded images in OUT_DIR'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    ImagenetDownloader(links_source=args.source, dest_dir=args.output).download_images(
        size=args.count, skip=args.skip
    )
    print("Done")
