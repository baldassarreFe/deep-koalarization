# *#dataset*

## Pipeline

All the steps are independent and persisted on the disk, the default and
recommended folder structure is:

```
~/imagenet
├── fall11_urls.txt
├── inception_resnet_v2_2016_08_30.ckpt
├── original
├── resized
├── filtered
└── tfrecords
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

#### Filtering

```bash
python3 -m dataset.filter <args>
```

Use `-h` to see the available options


#### Converting to TFRecords

```bash
python3 -O -m dataset.batch <args>
```

Passing `-c path/to/inception_resnet_v2_2016_08_30.ckpt` is highly recommended
over passing a url.

Omitting the `-O` will print all image pairs at the moment they are written to
a TFRecord. Given that we batch the inception operations, they will most likely
appear all at once when the batch gets written to disk.

Use `-h` to see the available options

## Space on disk notes

### The images

Out of the first 200 links, we get 131 valid images, that in their original
size take up a total of 17MB and then 2.5MB once resized to 299x299.

### The TFRecords

Originally, the images are stored using the jpeg compression, that makes their
size pretty small. On the other hand, once stored in a TFRecord they will simply
be in raw byte format and take up much more space.

Keep in mind that one example is made of:
- 2 images (299x299x3 uint8)
- 2 byte strings
- 1 embedding (1001 float32)

To save space we can use one of TFRecord compression options, or compress the
files after creation with a command like:

```
7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on "$RECORD.7z" "$RECORD"
```

Here is a comparison of the various compression options for a TFRecord:
- `tf.python_io.TFRecordCompressionType.NONE`
- `tf.python_io.TFRecordCompressionType.ZLIB`
- `tf.python_io.TFRecordCompressionType.GZIP`

|             |  NONE  |  ZLIB  |  GZIP  |
|-------------|-------:|-------:|-------:|
| Record size | 258 MB | 154 MB | 154 MB |
| After 7z    |  79 MB | 154 MB | 154 MB |

(tests made with the default value of 500 examples per tfrecord)
