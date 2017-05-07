import os
import sys
from os.path import join, isdir

from dataset.filtering import all_filters_with_base_args, filtered_filename
from dataset.shared import maybe_create_folder


class ImagenetFilters:
    def __init__(self, source_dir: str, dest_dir: str):
        if not isdir(source_dir):
            raise Exception('Input folder does not exists: {}'
                            .format(source_dir))
        self.source_dir = source_dir

        # Destination folder
        maybe_create_folder(dest_dir)
        self.dest_dir = dest_dir

    def filter_img(self, filename):
        source_file = join(self.source_dir, filename)
        for f in all_filters_with_base_args:
            dest_file = join(self.dest_dir,
                             filtered_filename(filename, f.__name__))
            try:
                f(source_file=source_file, dest_file=dest_file)
            except Exception as e:
                print(e, file=sys.stderr)

    def filter_all(self):
        for filename in os.listdir(self.source_dir):
            if os.path.isfile(os.path.join(self.source_dir, filename)):
                self.filter_img(filename)


# Run from the top folder as:
# python3 -m dataset.filter <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_originals, dir_filtered

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Apply filters to the images from one folder')
    parser.add_argument('-s', '--source-folder',
                        default=dir_originals,
                        type=str,
                        metavar='FOLDER',
                        dest='source',
                        help='use FOLDER as source of the images (default: {})'
                        .format(dir_originals))
    parser.add_argument('-o', '--output-folder',
                        default=dir_filtered,
                        type=str,
                        metavar='FOLDER',
                        dest='output',
                        help='use FOLDER as destination (default: {})'
                        .format(dir_filtered))

    args = parser.parse_args()
    ImagenetFilters(args.source, args.output).filter_all()
