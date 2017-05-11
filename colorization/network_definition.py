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


def _build_encoder():
    model = Sequential(name='encoder')
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
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
