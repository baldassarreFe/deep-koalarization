# Instructions

## Environment

The project is based on Python 3.6, to manage the dependencies contained in 
[`requirements.txt`](requirements.txt) a virtual environment is recommended.

```bash
python3.6 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Prior to training, the images from ImageNet need to be downloaded, resized and processed.

Extracting the embeddings in real time during training could potentially slow down the process, 
so the Inception embeddings are computed in advance, stored together with the images and fed directly to the fusion layer.   

At training time, for each image, we need the greyscale and colored versions and the inception embeddings. 
Storing the images and the embedding as separate files on disk (`jpeg/png` and `csv` for instance) would impact 
the performances during training, so all the image-embedding pairs are stored in binary format in large 
continuous [TFRecords](https://www.tensorflow.org/programmers_guide/datasets).

Refer to [DATASET](dataset/DATASET.md) for detailed instructions on data preparation.

## Training and evaluation

Before training, ensure that the folder `~/imagenet/tfrecords/` contains:
- training records as `lab_images_*.tfrecord`
- validation records as `val_lab_images_*.tfrecord` 
  (just rename some of the training records as validation, but do it before any training!)

The training script will train on all the training images, at the end of every epoch it will 
also checkpoint the weights and save to disk some colored images from the validation set. 

```bash
python3.6 -m colorization.train
```

The evaluation script will load the latest checkpoint, colorize images from the validation 
records and save them to disk. At the moment, it is not possible to operate on normal image
files (e.g. `jpeg` or `png`), but the images must be processed as TFRecords first.
```bash
python3.6 -m colorization.evaluate
```
