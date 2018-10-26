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

#from __future__ import division
  
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    LeakyReLU,
    add,
    merge,
    Add
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import (
    add,
    concatenate
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import keras

from keras.engine.topology import Layer
#import tensorflow.contrib.slim as slim
import tensorflow as tf


# Define the residual block as a new Layer
class Residual(Layer):
    def __init__(self, channels_in, kernel, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

    def call(self, x):
        # the residual block using Keras functional API
        '''
        first_layer = BatchNormalization()(x)
        x = Activation('relu')(first_layer)
        x = Conv2D(self.channels_in, self.kernel, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.channels_in, self.kernel, padding='same')(x)
        residual = Add()([x, first_layer])
        '''
        first_layer = Activation('linear', trainable=False)(x)
        x = Conv2D(self.channels_in, self.kernel, padding='same')(first_layer)
        x = Activation('relu')(x)
        x = Conv2D(self.channels_in, self.kernel, padding='same')(x)
        residual = Add()([x, first_layer])
        x = Activation('relu')(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


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


'''
def resUnit(input_layer, i):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = slim.batch_norm(input_layer, activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2, 64, [3, 3], activation_fn=None)
        part4 = slim.batch_norm(part3, activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5, 64, [3, 3], activation_fn=None)
        output = input_layer + part6
        return output
'''


'''
def colorizationResUnit(input_layer, i, model):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = BatchNormalization()(Input(shape=(None,None,1)))
        # model.add(part1)
        part2 = Activation('relu')(part1)
        # model.add(part2)
        part3 = Conv2D(64, (3, 3), activation=None, padding='same')(part2)
        # model.add(part3)
        part4 = BatchNormalization()(part3)
        # model.add(part4)
        part5 = Activation('relu')(part4)
        # model.add(part5)
        part6 = Conv2D(64, (3, 3), activation=None, padding='same')(part5)
        # model.add(part6)
        output = add([input_layer, part6])
        #output = input_layer + part6
        return output
'''


def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:

    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
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
    #y = shortcut + y
    #y = concatenate([shortcut, y])
    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = LeakyReLU()(y)
    #y = Conv2D(nb_channels_out, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    # model = Model(inputs=shortcut, outputs=y)
    return y#model


def _build_encoder():
    '''
    model = Sequential(name='encoder')
    input_layer = InputLayer(input_shape=(None, None, 1))
    model.add(input_layer)
    layer1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)
    model.add(layer1)
    #residual = colorizationResUnit(layer1, 0, model)
    #layer2 = add([layer1, residual])
    #model.add(layer2)
    # keras-resnet/resnet.py
    
    input_shape = (1, None, None)
    num_outputs = 256
    block_fn = basic_block
    repetitions = [2, 2, 2, 2]
    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

    # Permute dimension order if necessary
    if K.image_dim_ordering() == 'tf':
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

    # Load function from str if needed.
    block_fn = _get_block(block_fn)

    input = Input(shape=input_shape)
    conv1 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

    block = pool1
    filters = 64
    '''
    # tf-basics/simple-resnet/simple-resnet.py
    '''
    #input_layer = InputLayer(input_shape=(None, None, 1))
    input_layer = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32, name='input')
    model.add(input_layer)
    #layer1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)
    layer1 = slim.conv2d(input_layer, 64, [3, 3], stride=2, normalizer_fn=slim.batch_norm, scope='conv_'+str(0))
    model.add(layer1)
    for i in range(5):
        layer2 = resUnit(layer1, i)
        model.add(layer2)
        layer3 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)
        model.add(layer3)
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    '''
    # ????????????????????????????????????????????????
    '''
    model = Sequential(name='encoder')
    image_tensor = Input(shape=(224, 224, 1))
    input_layer = InputLayer(input_tensor=image_tensor)
    model.add(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)
    model.add(x)#(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')
    model.add(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)
    model.add(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')
    model.add(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)
    model.add(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')
    model.add(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')
    model.add(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')
    model.add(x)
    '''
    #image_tensor = Input(shape=(None, None, 1))
    #x = image_tensor#Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(image_tensor)
    #model.add(x)
    '''
    for i in range(3):
        project_shortcut = True if i == 0 else False
    '''
    #project_shortcut = False
    #shortcut = x
    #x = BatchNormalization()(x)
    #x = Activation('relu')(x)
    #y = Conv2D(64, (1, 1), activation=None, padding='same')(image_tensor)
    #x = BatchNormalization()(y)
    #x = Activation('relu')(y)
    #x = concatenate([image_tensor, y])
    #x = add([image_tensor, y])
    #x = add([shortcut, x])
    #x = Activation('relu')(x)
    #x = concatenate([shortcut, x])
    #x = shortcut + x
    # x = x(image_tensor)

    # x = Residual(64, (3, 3))
    # model.add(x)
    # x = residual_block(x, 64, 64)
    # x = colorizationResUnit(x, 0, None)
    # model.add(x)
    #model = Model(inputs=[image_tensor], outputs=[x])
    
    #res = Residual(64, (1, 1))
    #model.add(res)
   
    
    #model.add(Residual(64, (3, 3)))
    #model.add(Residual(128, (3, 3)))
    #first_layer = Activation('relu')(image_tensor)
    #y = Conv2D(64, (1, 1), activation=None, padding='same')(first_layer)
    #y = Activation('relu')(y)
    #y = Conv2D(64, (1, 1), activation=None, padding='same')(y)
    #x = add([first_layer, y])
    #x = Activation('relu')(x)
    #model.add(x)
    
    # Functional implementation
    image_tensor = Input(shape=(None, None, 1))#(224, 224, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(image_tensor)
    x = residual_block(x, 64, 64)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = residual_block(x, 128, 128)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = residual_block(x, 256, 256)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    model = Model(inputs=[image_tensor], outputs=[x])
    '''
    # Sequential Model implementation
    model = Sequential(name='encoder')
    image_tensor = Input(shape=(224, 224, 1))
    input_layer = InputLayer(input_tensor=image_tensor)
    model.add(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)
    model.add(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')
    model.add(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)
    model.add(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')
    model.add(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)
    model.add(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')
    model.add(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')
    model.add(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')
    model.add(x)
    return model
    '''
    return model#Sequential(layers=model.layers)


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


'''
def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier
'''
