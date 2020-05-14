"""Run module script

```
$ python3 -m dataset.resize <args>
```

Raises:
    Exception: [description]

"""


import argparse
from os import listdir
from os.path import join, isfile, isdir
from typing import Tuple

from PIL import Image
from resizeimage import resizeimage

from koalarization.dataset.shared import maybe_create_folder
    from koalarization.dataset.shared import DIR_ORIGINALS, DIR_RESIZED


class ImagenetResizer:
    """Class instance to resize the images."""

    def __init__(self, source_dir, dest_dir):
        """Constructor.

        Args:
            source_dir (str): Path to folder containing all original images.
            dest_dir (str): Path where to store resized images.

        Raises:
            Exception: If source_dir does not exist.

        """
        if not isdir(source_dir):
            raise Exception('Input folder does not exists: {}'
                            .format(source_dir))
        self.source_dir = source_dir

        # Destination folder
        maybe_create_folder(dest_dir)
        self.dest_dir = dest_dir

    def resize_img(self, filename, size=(299, 299)):
        """Resize image using padding.

        Resized image is stored in `dest_dir`.

        Args:
            filename (str): Filename of specific image.
            size (Tuple[int, int], optional): Output image shape. Defaults to (299, 299).

        """
        img = Image.open(join(self.source_dir, filename))
        orig_width, orig_height = img.size
        wanted_width, wanted_height = size
        ratio_w, ratio_h = wanted_width / orig_width, wanted_height / orig_height

        enlarge_factor = min(ratio_h, ratio_w)
        if enlarge_factor > 1:
            # Both sides of the image are shorter than the desired dimension,
            # so take the side that's closer in size and enlarge the image
            # in both directions to make that one fit
            enlarged_size = (int(orig_width * enlarge_factor), int(orig_height * enlarge_factor))
            img = img.resize(enlarged_size)

        # Now we have an image that's either larger than the desired shape
        # or at least one side matches the desired shape and we can resize
        # with contain
        res = resizeimage.resize_contain(img, size).convert('RGB')
        res.save(join(self.dest_dir, filename), res.format)

    def resize_all(self, size=(299, 299)):
        """Resizes all images within `source_dir`.

        Args:
            size (tuple, optional): Output image shape. Defaults to (299, 299).
        """
        for filename in listdir(self.source_dir):
            if isfile(join(self.source_dir, filename)):
                self.resize_img(filename, size)


def _parse_args():
    """Argparse setup.

    Returns:
        Namespace: Arguments.

    """
    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Resize images from a folder to 299x299.'
    )
    parser.add_argument('-s', '--source-folder',
                        default=DIR_ORIGINALS,
                        type=str,
                        metavar='FOLDER',
                        dest='source',
                        help='use FOLDER as source of the images (default: {})'
                        .format(DIR_ORIGINALS))
    parser.add_argument('-o', '--output-folder',
                        default=DIR_RESIZED,
                        type=str,
                        metavar='FOLDER',
                        dest='output',
                        help='use FOLDER as destination (default: {})'
                        .format(DIR_RESIZED))

    args = parser.parse_args()
    return args


if __name__ == '__main__':    
    args = _parse_args()
    ImagenetResizer(
        source_dir=args.source, 
        dest_dir=args.output
    ).resize_all(size=(299, 299))
