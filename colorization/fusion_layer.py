from keras.layers import RepeatVector, Reshape, concatenate, Conv2D


def fusion(img_encoded, img_emb):
    #  Get size of feature vector
    # TODO try with img_emb.shape
    inception_len = img_emb.get_shape().as_list()[1]
    # Get sides of the image encoded by the conv layers
    batch_size, width, height, depth = img_encoded.get_shape().as_list()

    # Repeat and reshape inception feature vector
    # (None, width * height, inception_len)
    img_emb = RepeatVector(width * height)(img_emb)

    # (None, width, height, inception_len)
    img_emb = Reshape((width, height, inception_len))(img_emb)

    # Concatenate inception with encode output
    # (None, width, height, inception_len + img_depth)
    concat = concatenate([img_emb, img_encoded], axis=3)
    # (None, width, height, fusion_out_depth)
    fusion_out_depth = 256
    fusion_out = Conv2D(fusion_out_depth, (1, 1), activation='relu')(concat)
    return fusion_out
