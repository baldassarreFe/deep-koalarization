from os import makedirs
from os.path import expanduser, join

# Default folders
dir_root = join(expanduser('~'), 'imagenet')
dir_originals = join(dir_root, 'original')
dir_resized = join(dir_root, 'resized')
dir_filtered = join(dir_root, 'filtered')
dir_tfrecord = join(dir_root, 'tfrecords')


def maybe_create_folder(folder):
    makedirs(folder, exist_ok=True)
