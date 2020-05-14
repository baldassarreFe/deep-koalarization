import itertools
from os import makedirs
from os.path import expanduser, join

# Default folders
DIR_ROOT = join(expanduser('~'), 'imagenet')
DIR_ORIGINALS = join(DIR_ROOT, 'original')
DIR_RESIZED = join(DIR_ROOT, 'resized')
DIR_TFRECORD = join(DIR_ROOT, 'tfrecords')
DIR_METRICS = join(DIR_ROOT, 'metrics')
DIR_CHECKPOINTS = join(DIR_ROOT, 'checkpoints')
FILE_IMAGEURLS = join(DIR_ROOT, 'imagenet_fall11_urls.txt')
# 'http://media.githubusercontent.com/media/akando42/1stPyTorch/master/fall11_urls.txt'

def maybe_create_folder(folder):
    makedirs(folder, exist_ok=True)


def progressive_filename_generator(pattern='file_{}.ext'):
    for i in itertools.count():
        yield pattern.format(i)
