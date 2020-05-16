import itertools
from os import makedirs

def maybe_create_folder(folder):
    makedirs(folder, exist_ok=True)


def progressive_filename_generator(pattern="file_{}.ext"):
    for i in itertools.count():
        yield pattern.format(i)
