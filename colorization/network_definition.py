"""
Given the L channel of an Lab image (range [-1, +1]), output a prediction over
the a and b channels in the range [-1, 1].
In the neck of the conv-deconv network use the features from a feature extractor
(e.g. Inception) and fuse them with the conv output.
"""

from keras.engine import InputLayer
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential

from colorization.fusion_layer import FusionLayer
  
from keras.models import Model
from keras.layers import (
    Input,
    LeakyReLU,
    ReLU,
    Dropout
)
from keras.layers.merge import (
    add
)
from keras.layers.normalization import BatchNormalization
import tensorflow as tf


class Colorization:
    def __init__(self, depth_after_fusion):
        self.encoder = _build_encoder()
        self.fusion = FusionLayer()
        self.after_fusion = Conv2D(
            depth_after_fusion, (1, 1), activation='relu')
        self.decoder = _build_decoder(depth_after_fusion)

    def build(self, img_l, img_emb):
        img_enc = self.encoder(img_l)

        fusion = self.fusion([img_enc, img_emb])
        fusion = self.after_fusion(fusion)

        return self.decoder(fusion)


def wideResUnit(y, nb_channels_in, nb_channels_out, name):
    with tf.name_scope(name):
        shortcut = y
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = Dropout(0.5)(y)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = add([shortcut, y])
        return y


def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def residual_block(y, nb_channels_in, nb_channels_out, name, _strides=(1, 1), _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:

    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    with tf.name_scope(name):
        shortcut = y
        # we modify the residual building block as a bottleneck design to make the network more economical
        y = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)
        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = add_common_layers(y)
        y = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = BatchNormalization()(y)
        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        y = add([shortcut, y])
        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = LeakyReLU()(y)
        return y


def _build_encoder():    
    # Functional implementation
    image_tensor = Input(shape=(None, None, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(image_tensor)
    #x = wideResUnit(x, 64, 64, 'res1')
    # x = residual_block(x, 64, 64, 'res1')
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = wideResUnit(x, 128, 128, 'res2')
    # x = residual_block(x, 128, 128, 'res2')
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # x = wideResUnit(x, 256, 256, 'res3')
    # x = residual_block(x, 256, 256, 'res3')
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    model = Model(inputs=[image_tensor], outputs=[x])
    return model


def _build_decoder(encoding_depth):
    model = Sequential(name='decoder')
    model.add(InputLayer(input_shape=(None, None, encoding_depth)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))
    return model
