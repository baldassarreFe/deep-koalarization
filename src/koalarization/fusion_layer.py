from keras import backend as K
from keras.engine import Layer


class FusionLayer(Layer):
    def call(self, inputs, mask=None):
        imgs, embs = inputs
        reshaped_shape = imgs.shape[:3].concatenate(embs.shape[1])
        embs = K.repeat(embs, imgs.shape[1] * imgs.shape[2])
        embs = K.reshape(embs, reshaped_shape)
        return K.concatenate([imgs, embs], axis=3)

    def compute_output_shape(self, input_shapes):
        # Must have 2 tensors as input
        assert input_shapes and len(input_shapes) == 2
        imgs_shape, embs_shape = input_shapes

        # The batch size of the two tensors must match
        assert imgs_shape[0] == embs_shape[0]

        # (batch_size, width, height, embedding_len + depth)
        return imgs_shape[:3] + (imgs_shape[3] + embs_shape[1],)
