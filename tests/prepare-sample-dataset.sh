echo "1/3 Downloading unsplash images"
python -m koalarization.dataset.download './data/unsplash.txt' './tests/data/original/'
echo "2/3 Resizing images"
python -m koalarization.dataset.resize './tests/data/original' './tests/data/resized/'
echo "3/3 Generating TF Records"
python -m koalarization.dataset.lab_batch -c './tests/data/inception_resnet_v2_2016_08_30.ckpt'\
    './tests/data/resized' './tests/data/tfrecords'