# Import libraries to list and create directories
import os 
# Tensorflow 
import tensorflow as tf
# Inception
#from inception_preprocessing import preprocess_for_eval
# Numerical libraries
import numpy as np
# Others
import urllib
import urllib.request
import tarfile


# Download checkpoint for inception-resnet-v2 if not already downloaded
def inception_resnet_v2_maybe_download(self, checkpoint_dir):
	checkpoint_file = os.join(checkpoint_dir, "inception_resnet_v2_2016_08_30.ckpt")
    if not os.path.exists(checkpoint_file):
        if not os.path.exists(self.checkpoint_file+'.tar.gz'):
            file_url= 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
            print("Downloading checkpoint...")
            urllib.request.urlretrieve(file_url, self.checkpoint_file+'.tar.gz')
            r = requests.get(file_url)
        print("Extracting checkpoint...")
        tar = tarfile.open(self.checkpoint_file+'.tar.gz')
        tar.extractall()
        tar.close()
    else:
        print("> Checkpoint already downloaded and extracted!")

	# Preprocess data (zero mean, range)
def preprocessing(self, input_tensor):
	scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
	scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
	scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
	return scaled_input_tensor
		