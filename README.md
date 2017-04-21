# *#nofilter*
#### Reversing image filters using Image-to-Image translation techniques
This project is currently being developed as part of the [DD2424 Deep Learning in Data Science course](https://www.kth.se/student/kurser/kurs/DD2424?l=en) at [KTH Royal Institute of Technology](http://kth.se), spring 2017.

| Author               | GitHub                                            |
|:---------------------|:--------------------------------------------------|
| Federico Baldassarre | [baldassarreFe](https://github.com/baldassarreFe) |
| Diego Gonzalez Morin | [diegomorin8](https://github.com/diegomorin8)     |
| Lucas Rodés Guirao   | [lucasrodes](https://github.com/lucasrodes)       |

## Problem
We got the inspiration from the work of Richard Zhang, Phillip Isola and Alexei A. Efros, who realized a network able to colorize black and white images ([blog post](http://richzhang.github.io/colorization/) and [paper](https://arxiv.org/abs/1603.08511)). They trained a network on ImageNet pictures preprocessed to make them grayscale, with the colored image as the output target.


Then we also saw the experiments of Satoshi Izuka, Edgar Simo-Serra and Hiroshi Ishikawa, who added image classification features to raw pixels fed to the network, improving the overall results ([YouTube review](https://www.youtube.com/watch?v=MfaTOXxA8dM), [blog post](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/) and [paper](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf)).

Our idea is to apply a similar technique to overly filtered images (e.g. Instagram images), i.e. going from the modified color image to a version that is closer to the original, using both the raw pixels and object recognition features.

We plan to enrich the pixel information of each image with semantic information about its content. However, object recognition is already a complex problem and would be worth a project on its own.
For this reason we are not going to build an image classifier from scratch. Instead we will use a network that has already been trained on a large variety of images, such as Google’s Inception model or AlexNet, both trained on ImageNet. Adapting pre trained models and using them as feature extractor for other tasks is the base concept of transfer learning.

![Inception v3](images/inception_v3.png)

The hidden layers of these model have learned to create a semantic representation of the image that is then used by the final layer (fully connected + softmax) to label the objects in the image. By “cutting” the model at one of its final layers we will get a high dimensional representation of image features, that will be used by our network to perform the un-filtering task (TensorFlow [tutorial](https://www.tensorflow.org/tutorials/image_retraining) on transfer learning, another [tutorial](https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html) and arXiv [paper](https://arxiv.org/abs/1403.6382)).

## Training data
Training data for this experiment could actually come from every source, but we are thinking of using images from ImageNet and apply instagram-like filters. The motivation for this choice would be to use a dataset that is nowadays considered the de-facto reference for image tasks. It will also make our experiments easily reproducible, compared to using our own images.

After the network is trained, if we have time, we might build a plugin to use actual pictures from Instagram.

## Software packages
We will be working with Python 3, for it is widely used in the data science and machine learning communities.

As for additional libraries, for machine learning, deep learning and image manipulation we will use TensorFlow, NumPy/SciPy, Scikit, Pillow, imagemagick etc.

## Experiments
We are going to start small, with fewer images and fewer filter styles. Once the technique seems to be working we are going to expand the dataset and the filters, adapting also the size of the network.

Finally we will test the network against real images taken from Instagram.

## Evaluation
In this project, we deal with a regression problem, where we want to estimate how the original picture looked like given a filtered version of it. The regression task is performed by a Deep Network and can be expressed as a function _F_.


Furthermore, since our task is supervised learning, for each sample we have two images. In particular, we need the filtered and the original version, i.e. X and Y, respectively. After training, the network should be able to estimate the original image given its filtered version, i.e. _Y= F(X)_.

In order to find an optimal model parameter setting, we need a way to measure the similarity between the estimated image and the original image, i.e. we need to define a loss function

_L{F(X), Y} = L{Y, Y}._

To evaluate the performance of our model, we need to have some measure of how good it performs on a test set. Some techniques are Explained Variance Score, Mean Absolute Error, Mean Squared Error, Median Absolute Error and R2 Score. In the final report, we will explain which technique we have used and why we did so.
