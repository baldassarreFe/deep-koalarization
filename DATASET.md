# Dataset

Prior to training, the images from ImageNet need to be downloaded, resized and processed.

Extracting the embeddings in real time during training could potentially slow down the process, 
so the Inception embeddings are computed in advance, stored together with the images and fed directly to the fusion layer.   

At training time, for each image, we need the greyscale and colored versions and the inception embeddings. 
Storing the images and the embedding as separate files on disk (`jpeg/png` and `csv` for instance) would impact 
the performances during training, so all the image-embedding pairs are stored in binary format in large 
continuous [TFRecords](https://www.tensorflow.org/programmers_guide/datasets).

## Pipeline

All the data preparation steps are independent and persisted on the disk, the default (and recommended) folder structure is:

```
./data
├── fall11_urls.txt
├── imagenet1000_clsid_to_human.pkl
├── inception_resnet_v2_2016_08_30.ckpt
├── original
├── resized
└── tfrecords
```

### Imagenet images
To download the images from ImageNet, we provide a script that takes as input a file containing image URLs (one per line).
You are not restricted in any way to use images from ImageNet, you can provide any list of URLs to the script and it will take care of the download.
See [unsplash.txt](data/unsplash.txt) for an example.
Also, if you already have a collection of images for training, place them in `data/original` and skip to the next step.

> **Note:**
> There used to be a [file](http://www.image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz) 
> containing the image URLs for ImageNet 2011 available without registration on the 
> [official website](http://image-net.org/download-imageurls).
> Since the link appears to be down, you may want to use this 
> [non-official file](http://github.com/akando42/1stPyTorch/blob/master/fall11_urls.txt) instead.

```bash
wget -O 'data/imagenet_fall11_urls.txt' 'https://github.com/akando42/1stPyTorch/raw/master/fall11_urls.txt'
python -m koalarization.dataset.download 'data/imagenet_fall11_urls.txt' 'data/original'
```

The download script also accepts a URL as `source`, but downloading the URL file separately 
and passing it as `path/to/urls.txt` is highly recommended. 
Use `-h` to see all available options.

### Resizing for training
To be able to train in batches, we resize all images to `299x299`. 
Use the following script to achieve this:

```bash
python -m koalarization.dataset.resize 'data/original' 'data/resized'
```

Use `-h` to see the available options

### Converting to TFRecords
First download the pretrained Inception model for feature extraction, then use the `lab_batch` script to process all images from the resized folder:
```bash
wget -O - 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz' | tar -xzv -C 'data'
python -m koalarization.dataset.lab_batch -c 'data/inception_resnet_v2_2016_08_30.ckpt' 'data/resized' 'data/tfrecords'
```

If `-c` is omitted the script will download the checkpoint by itself, however downloading the checkpoint separately is highly recommended.
Use `-h` to see the available options.

### Validation set
Some tfrecords are selected to be used as a validation set. This is done by simply renaming, for example:
```bash
mv 'data/tfrecord/lab_images_0.tfrecord' 'data/tfrecord/val_lab_images_0.tfrecord'
```

## Space on disk notes

### The images
Out of the first 200 links, we get 131 valid images, that in their original
size take up a total of 17MB and then 2.5MB once resized to 299x299.

### The TFRecords
Originally, the images are stored using the `jpeg` compression, that makes their
size pretty small. On the other hand, once stored in a TFRecord they will simply
be in raw byte format and take up much more space.

Keep in mind that one example is made of:
- 1 image `L` (299x299x1 float32)
- 1 image `a*b*` (299x299x2 float32)
- 1 embedding (1001 float32)
- 1 byte string

To save space we can use one of TFRecord compression options, or compress the
files after creation with a command like:

```bash
RECORD='data/tfrecord/lab_images_0.tfrecord'
7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on "$RECORD.7z" "$RECORD"
```
