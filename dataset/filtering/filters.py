import cv2
from PIL import Image

from .cv2_ops import hue_saturation, brightness_contrast, channel_enhance
from .imagemagik_ops import border, colortone, vignette, execute


def filtered_filename(filename, filtername):
    name, ext = filename.split('.')
    return '{}_{}.{}'.format(name, filtername, ext)


"""
    OpenCV based filters are applied in memory to an image,
    so we wrap them to make them similar to the imagemagick ones
"""


class OpenCvWrapper(object):
    def __init__(self, f):
        self.f = f
        self.__name__ = f.__name__
        self.__doc__ = f.__doc__

    def __call__(self, **kwargs):
        if 'source_file' in kwargs and 'dest_file' in kwargs:
            source_file = kwargs['source_file']
            dest_file = kwargs['dest_file']

            del kwargs['source_file']
            del kwargs['dest_file']

            img = cv2.imread(source_file)
            img = self.f(img=img, **kwargs)
            cv2.imwrite(dest_file, img)
        elif 'img' in kwargs:
            return self.f(**kwargs)
        else:
            raise Exception(
                'Call {} with either an image or a source_file and dest_file'
                    .format(self.f.__name__))


@OpenCvWrapper
def nashville(img, hue=1, saturation=1.5, contrast=1.5, brightness=-30):
    img = hue_saturation(img, hue, saturation)
    img = brightness_contrast(img, contrast, brightness)
    return img


@OpenCvWrapper
def gotham(img, hue=1, saturation=0.1, contrast=1.3, brightness=-20):
    img = hue_saturation(img, hue, saturation)
    img = brightness_contrast(img, contrast, brightness)
    return img


@OpenCvWrapper
def claredon(img, hue=1.2, saturation=1.4, contrast=1.4, brightness=- 20):
    img = hue_saturation(img, hue, saturation)
    img = brightness_contrast(img, contrast, brightness)
    img = channel_enhance(img, "B", 1.3)
    return img


"""
    Imagemagick based filters are applied in place on the image.
    The wrapper is used to save the image before converting
"""


def lomo(source_file, dest_file):
    execute(
        "convert {source} -channel R -level 33% -channel G -level 33% {dest}"
            .format(
            source=source_file,
            dest=dest_file
        )
    )
    vignette(dest_file, dest_file)


def kelvin(source_file, dest_file):
    width, height = Image.open(source_file).size
    execute(
        ("convert \( {source} -auto-gamma -modulate 120,50,100 \) " +
         "\( -size {width}x{height} -fill 'rgba(255,153,0,0.5)' " +
         "-draw 'rectangle 0,0 {width},{height}' \) -compose multiply {dest}")
            .format(
            source=source_file,
            dest=dest_file,
            width=width,
            height=height
        )
    )


def nash2(source_file, dest_file):
    colortone(source_file, dest_file, '#222b6d', 50, 0)
    colortone(dest_file, dest_file, '#f7daae', 120, 1)
    execute(
        'convert {source} -contrast -modulate 100,150,100 -auto-gamma {dest}'
            .format(
            source=dest_file,
            dest=dest_file
        )
    )


def toaster(source_file, dest_file):
    colortone(source_file, dest_file, '#330000', 50, 0)
    execute(
        ('convert {source} -modulate 150,80,100 -gamma 1.2 ' +
         '-contrast -contrast {dest}')
            .format(
            source=dest_file,
            dest=dest_file
        )
    )
    vignette(dest_file, dest_file, 'none', 'LavenderBlush3')
    vignette(dest_file, dest_file, '#ff9966', 'none')
    border(dest_file, dest_file, 'white')


all_filters_with_base_args = [
    nashville,
    gotham,
    claredon,
    lomo,
    kelvin,
    nash2,
    toaster
]
