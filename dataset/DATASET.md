# Dataset

Prior to training, the images from ImageNet need to be downloaded, resized and processed.

Extracting the embeddings in real time during training could potentially slow down the process, 
so the Inception embeddings are computed in advance, stored together with the images and fed directly to the fusion layer.   

At training time, for each image, we need the greyscale and colored versions and the inception embeddings. 
Storing the images and the embedding as separate files on disk (`jpeg/png` and `csv` for instance) would impact 
the performances during training, so all the image-embedding pairs are stored in binary format in large 
continuous [TFRecords](https://www.tensorflow.org/programmers_guide/datasets).

## Pipeline

All the data preparation steps are independent and persisted on the disk, the default and
recommended folder structure is:

```
~/imagenet
├── fall11_urls.txt
├── imagenet1000_clsid_to_human.pkl
├── inception_resnet_v2_2016_08_30.ckpt
├── original
├── resized
└── tfrecords
```

#### Imagenet labels

```bash
wget https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl
```

#### Getting the images from Imagenet

```bash
python3 -m dataset.download <args>
```

Passing `-s path/to/fall11_urls.txt` is highly recommended over passing a url.

Use `-h` to see the available options

#### Resizing

```bash
python3 -m dataset.resize <args>
```

Use `-h` to see the available options

#### Converting to TFRecords

```bash
python3 -O -m dataset.lab_batch <args>
```

Passing `-c path/to/inception_resnet_v2_2016_08_30.ckpt` is highly recommended
over passing a url. To download the checkpoint it separately:

```bash
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
tar -xvf inception_resnet_v2_2016_08_30.tar.gz
```

Omitting the `-O` (optimize) will print all image names at the moment they are written to
a TFRecord. These prints will most likely appear all at once, 
after TensorFlow has written the batch on disk and passes the control back to Python.

Use `-h` to see the available options

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

```
7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on "$RECORD.7z" "$RECORD"
```
