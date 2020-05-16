# Instructions

## Environment
The project is based on Python 3.6, to manage the dependencies contained in 
[`requirements.txt`](requirements.txt) a virtual environment is recommended.

```bash
python3.6 -m venv venv
source venv/bin/activate
pip install -e .
```

Even better, use a [Conda environment](https://docs.conda.io/):
```bash
conda create -y -n koalarization python=3.6
conda activate koalarization
pip install -e .
```

For GPU-support, run:

```
$ pip install -e .[gpu]
```

## Dataset
Prior to training, the images from ImageNet need to be downloaded, resized and processed.

Extracting the embeddings in real time during training could potentially slow down the process, 
so the Inception embeddings are computed in advance, stored together with the images and fed directly to the fusion layer.   

At training time, for each image, we need the greyscale and colored versions and the inception embeddings. 
Storing the images and the embedding as separate files on disk (`jpeg/png` and `csv` for instance) would impact 
the performances during training, so all the image-embedding pairs are stored in binary format in large 
continuous [TFRecords](https://www.tensorflow.org/programmers_guide/datasets).

Refer to [DATASET](DATASET.md) for detailed instructions on data preparation.

## Training and evaluation

Before training, ensure that the folder `data` contains:
- training records as `tfrecords/lab_images_*.tfrecord`
- validation records as `tfrecords/val_lab_images_*.tfrecord` 
  (just rename some training records as validation, but do it before any training!)

The training script will train on all the training images, and regularly 
checkpoint the weights and save to disk some colored images from the validation set.

All training logs, metrics and checkpoints are saved in `runs/run_id`.

```bash
python -m koalarization.train \
  --run-id 'run1' \
  --train-steps 100 \
  --val-every 20 \
  'data/tfrecords' 'runs/'
```

The evaluation script will load the latest checkpoint, colorize images from the validation 
records and save them to disk. At the moment, it is not possible to operate on normal image
files (e.g. `jpeg` or `png`), but the images must be processed as TFRecords first.
```bash
python -m koalarization.evaluate \
  --run-id 'run1' \
  'data/tfrecords' 'runs/'
```
