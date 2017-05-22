# Deep Koalarization
#### Image colorization using deep CNN and features extracted from Inception-ResNet-v2

This project was developed as part of the [DD2424 Deep Learning in Data Science course](https://www.kth.se/student/kurser/kurs/DD2424?l=en) at [KTH Royal Institute of Technology](http://kth.se), spring 2017.

We attach the full [report](report.pdf) and our [slides](slides.pdf).

| Author               | GitHub                                            |
|:---------------------|:--------------------------------------------------|
| Federico Baldassarre | [baldassarreFe](https://github.com/baldassarreFe) |
| Diego Gonzalez Morin | [diegomorin8](https://github.com/diegomorin8)     |
| Lucas Rodés Guirao   | [lucasrodes](https://github.com/lucasrodes)       |

## Intro
We got the inspiration from the work of Richard Zhang, Phillip Isola and Alexei A. Efros, who realized a network able to colorize black and white images ([blog post](http://richzhang.github.io/colorization/) and [paper](https://arxiv.org/abs/1603.08511)). They trained a network on ImageNet pictures preprocessed to make them gray-scale, with the colored image as the output target.

Then we also saw the experiments of Satoshi Iizuka, Edgar Simo-Serra and Hiroshi Ishikawa, who added image classification features to raw pixels fed to the network, improving the overall results ([YouTube review](https://www.youtube.com/watch?v=MfaTOXxA8dM), [blog post](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) and [paper](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf)).

## Network
In our work we implement a similar architecture, enriching the pixel information of each image with semantic information about its content. However, object recognition is already a complex problem and would be worth a project on its own.
For this reason we are not going to build an image classifier from scratch. Instead we will use a network that has already been trained on a large variety of images, such as Google’s [Inception-ResNet-v2](https://arxiv.org/abs/1602.07261), which is trained on ImageNet.

![Inception v3](assets/our_net.png)

The hidden layers of these models learned to create a semantic representation of the image that is then used by the final layer (fully connected + softmax) to label the objects in the image. By “cutting” the model at one of its final layers we will get a high dimensional representation of image features, that will be used by our network to perform the colorization task (TensorFlow [tutorial](https://www.tensorflow.org/tutorials/image_retraining) on transfer learning, another [tutorial](https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html) and arXiv [paper](https://arxiv.org/abs/1403.6382)).

The _fusion_ between the fixed-size embedding and the intermediary result of the convolutions is performed by means of replication and stacking as described in the original paper:

![Fusion](assets/fusion_layer.png)

We have used the MSE loss as the objective function.

## Dataset
Training data for this experiment could actually come from every source, but we rather prefer using images from [ImageNet](http://www.image-net.org), which nowadays is considered the de-facto reference for image tasks. It will also make our experiments easily reproducible, compared to using our own images.

## Software packages
Our project is implemented in Python 3, for it is widely used in the data science and machine learning communities. As for additional libraries, for machine learning, deep learning and image manipulation we rely on Keras/TensorFlow, NumPy/SciPy, Scikit. Refer to [requirements.txt](requirements.txt) for a complete list of the dependencies.

## Results

#### ImageNet

![ImageNet 1](assets/comparison.png)

#### Historical Pictures

![Historical 1](assets/historical.png)
