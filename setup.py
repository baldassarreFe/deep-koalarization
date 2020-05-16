"""setup script"""
from setuptools import setup, find_packages
import os
import glob

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

with open(os.path.join(this_directory, 'requirements2.txt')) as f:
    requirements = f.readlines()

setup(
    name='deep-koalarization',
    version="0.1.0",
    description="Keras/Tensorflow implementation of our paper Grayscale Image Colorization using deep CNN and "
                "Inception-ResNet-v2",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/baldassarreFe/deep-koalarization',
    author='Federico Baldassare, Diego González Morín, Lucas Rodés-Guirao',
    license='GPL-v3',
    install_requires=requirements,
    extras_require={"gpu": ['tensorflow-gpu==1.3.0']},
    packages=find_packages('src'),
    package_dir={'': 'src'},
    namespace_packages=['koalarization'],
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6"
    ],
    keywords='Image colorization using Deep Learning CNNs',
    project_urls={
        'Website': 'https://lcsrg.me/deep-koalarization',
        'Github': 'http://github.com/baldassarreFe/deep-koalarization'
    },
    python_requires='>=3.5'
)