import os

root = 'imagenet'
dir_originals = os.path.join(root, 'original')
dir_resized = os.path.join(root, 'resized')
dir_filtered = os.path.join(root, 'filtered')


def maybe_create_folder(folder):
    os.makedirs(folder, exist_ok=True)
