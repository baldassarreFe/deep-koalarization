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
python3 -m dataset.filters <args>
```

Use `-h` to see the available options


#### Converting to TFRecords

```bash
python3 -m dataset.filters <args>
```

Passing `-c path/to/inception_resnet_v2_2016_08_30.ckpt` is highly recommended
over passing a url.

Use `-h` to see the available options

## Notes

Out of the first 200 links, we get 131 valid images, that initially take
a total of 17MB and then 2.5MB once resized.

Using the default value of 500 examples per tfrecord results in chunks of
roughly 258MB each that shrink to 67MB if compressed with:

```
7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on "$RECORD.7z" "$RECORD"
```