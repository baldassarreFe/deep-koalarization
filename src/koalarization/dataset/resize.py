"""Resizing the images for the model

To be able to train in batches, we resize all images to a common shape (299x299).
Use the following script to achieve this:

```
python3 -m koalarization.dataset.resize path/to/original path/to/resized
```

Use `-h` to see the available options
"""


import argparse
from os import listdir
from os.path import join, isfile, isdir

from PIL import Image
from resizeimage import resizeimage

from .shared import maybe_create_folder


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
            raise Exception("Input folder does not exists: {}".format(source_dir))
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
            enlarged_size = (
                int(orig_width * enlarge_factor),
                int(orig_height * enlarge_factor),
            )
            img = img.resize(enlarged_size)

        # Now we have an image that's either larger than the desired shape
        # or at least one side matches the desired shape and we can resize
        # with contain
        res = resizeimage.resize_contain(img, size).convert("RGB")
        res.save(join(self.dest_dir, filename), res.format)

    def resize_all(self, size=(299, 299)):
        """Resizes all images within `source_dir`.

        Args:
            size (tuple, optional): Output image shape. Defaults to (299, 299).
        """
        for filename in listdir(self.source_dir):
            img_path = join(self.source_dir, filename)
            if filename.endswith((".jpg", ".jpeg")) and isfile(img_path):
                self.resize_img(filename, size)


def _parse_args():
    """Argparse setup.

    Returns:
        Namespace: Arguments.

    """

    def size_tuple(size: str):
        size = tuple(map(int, size.split(",", maxsplit=1)))
        if len(size) == 1:
            size = size[0]
            size = (size, size)
        return size

    parser = argparse.ArgumentParser(
        description="Resize all images in a folder to a common size."
    )
    parser.add_argument(
        "source", type=str, metavar="SRC_DIR", help="resize all images in SRC_DIR"
    )
    parser.add_argument(
        "output", type=str, metavar="OUR_DIR", help="save resized images in OUR_DIR"
    )
    parser.add_argument(
        "-s --size",
        type=size_tuple,
        default=(299, 299),
        metavar="SIZE",
        dest="size",
        help="resize images to SIZE, can be a single integer or two comma-separated (W,H)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    ImagenetResizer(source_dir=args.source, dest_dir=args.output).resize_all(
        size=args.size
    )
    print("Done")
