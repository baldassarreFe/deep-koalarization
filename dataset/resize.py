from os import listdir
from os.path import join, isfile, isdir
from typing import Tuple

import numpy as np
from PIL import Image
from resizeimage import resizeimage

from dataset.shared import maybe_create_folder


class ImagenetResizer:
    def __init__(self, source_dir: str, dest_dir: str):
        if not isdir(source_dir):
            raise Exception('Input folder does not exists: {}'
                            .format(source_dir))
        self.source_dir = source_dir

        # Destination folder
        maybe_create_folder(dest_dir)
        self.dest_dir = dest_dir

    def resize_img(self, filename: str, size: Tuple[int, int] = (299, 299)):
        """
        Resizes the image using padding
        :param filename:
        :param size:
        """
        img = Image.open(join(self.source_dir, filename))
        width, height = img.size
        orig_shape = np.array(img.size)
        wanted_shape = np.array(size)
        ratios = wanted_shape / orig_shape
        wanted_width, wanted_height = size
        ratio_w, ratio_h = wanted_width / width, wanted_height / height

        if np.alltrue(ratios > 1):
            # Both sides of the image are shorter than the desired dimension,
            # so take the side that's closer in size and enlarge the image
            # in both directions to make that one fit
            factor = min(ratio_h, ratio_w)
            img = img.resize((int(width * factor), int(height * factor)))

        # Now we have an image that's either larger than the desired shape
        # or at least one side matches the desired shape and we can resize
        # with contain
        cover = resizeimage.resize_contain(img, size)
        cover.save(join(self.dest_dir, filename), 'JPEG')

    def resize_all(self, size=(299, 299)):
        for filename in listdir(self.source_dir):
            if isfile(join(self.source_dir, filename)):
                self.resize_img(filename, size)


# Run from the top folder as:
# python3 -m dataset.resize <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_originals, dir_resized

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Resize images from a folder to 299x299')
    parser.add_argument('-s', '--source-folder',
                        default=dir_originals,
                        type=str,
                        metavar='FOLDER',
                        dest='source',
                        help='use FOLDER as source of the images (default: {})'
                        .format(dir_originals))
    parser.add_argument('-o', '--output-folder',
                        default=dir_resized,
                        type=str,
                        metavar='FOLDER',
                        dest='output',
                        help='use FOLDER as destination (default: {})'
                        .format(dir_resized))

    args = parser.parse_args()
    ImagenetResizer(args.source, args.output).resize_all((299, 299))
