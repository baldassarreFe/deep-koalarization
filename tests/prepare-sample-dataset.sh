# If following paths are changed, make sure to change scripts in ./tests/batching/ accordingly
UNSPASH_URLS='./data/unsplash.txt' 
DIR_ORIGINAL='./tests/data/original/'
DIR_RESIZED='./tests/data/resized/'
DIR_TFRECORDS='./tests/data/tfrecords'
CHECKPOINT_INCEPTION='./data/inception_resnet_v2_2016_08_30.ckpt'

echo "1/3 Downloading unsplash images"
python -m koalarization.dataset.download ${UNSPASH_URLS} ${DIR_ORIGINAL}
echo "2/3 Resizing images"
python -m koalarization.dataset.resize ${DIR_ORIGINAL} ${DIR_RESIZED}
echo "3/3 Generating TF Records"  # Assumes inception checkpoint is placed in ./data
python -m koalarization.dataset.lab_batch -c ${CHECKPOINT_INCEPTION}\
    ${DIR_RESIZED} ${DIR_TFRECORDS}