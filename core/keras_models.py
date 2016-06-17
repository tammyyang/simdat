import h5py
from collections import OrderedDict
from keras import regularizers
from keras.models import Sequential
from keras.models import Graph
from keras.models import Model
from keras.layers import Input, Activation, merge
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D


def SqueezeNet(nb_classes, inputs=(3, 224, 224), weights_path=None):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)

    @param nb_classes: total number of final categories

    Arguments:
    inputs -- shape of the input images (channel, cols, rows)

    """

    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        96, 7, 7, activation='relu', init='glorot_uniform',
        subsample=(2, 2), border_mode='same', name='conv1')(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

    fire2_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_squeeze')(maxpool1)
    fire2_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand2')(fire2_squeeze)
    merge2 = merge(
        [fire2_expand1, fire2_expand2], mode='concat', concat_axis=1)

    fire3_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_squeeze')(merge2)
    fire3_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand2')(fire3_squeeze)
    merge3 = merge(
        [fire3_expand1, fire3_expand2], mode='concat', concat_axis=1)

    fire4_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_squeeze')(merge3)
    fire4_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand2')(fire4_squeeze)
    merge4 = merge(
        [fire4_expand1, fire4_expand2], mode='concat', concat_axis=1)
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

    fire5_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_squeeze')(maxpool4)
    fire5_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand2')(fire5_squeeze)
    merge5 = merge(
        [fire5_expand1, fire5_expand2], mode='concat', concat_axis=1)

    fire6_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_squeeze')(merge5)
    fire6_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand2')(fire6_squeeze)
    merge6 = merge(
        [fire6_expand1, fire6_expand2], mode='concat', concat_axis=1)

    fire7_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_squeeze')(merge6)
    fire7_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand1')(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand2')(fire7_squeeze)
    merge7 = merge(
        [fire7_expand1, fire7_expand2], mode='concat', concat_axis=1)

    fire8_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_squeeze')(merge7)
    fire8_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand1')(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand2')(fire8_squeeze)
    merge8 = merge(
        [fire8_expand1, fire8_expand2], mode='concat', concat_axis=1)

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)

    fire9_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_squeeze')(maxpool8)
    fire9_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand1')(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand2')(fire9_squeeze)
    merge9 = merge(
        [fire9_expand1, fire9_expand2], mode='concat', concat_axis=1)

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        nb_classes, 1, 1, init='glorot_uniform',
        border_mode='valid', name='conv10')(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D((13, 13), name='avgpool10')(conv10)
    # avgpool10 = AveragePooling2D((1, 1), name='avgpool10')(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    softmax = Activation("softmax", name='softmax')(flatten)

    model = Model(input=input_img, output=softmax)
    if weights_path is not None:
        load_weights(model, weights_path)

    return model


def Inception_v3(weights_path=None,
                 DIM_ORDERING='th', WEIGHT_DECAY=0,
                 USE_BN=False, NB_CLASS=1000):

    from keras.layers import Input, merge

    def conv2D_bn(x, nb_filter, nb_row, nb_col,
                  border_mode='same', subsample=(1, 1),
                  activation='relu', batch_norm=USE_BN,
                  weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):

        '''Utility function to apply to a tensor a module conv + BN
        with optional weight decay (L2 weight regularization).
        '''

        if weight_decay:
            W_regularizer = regularizers.l2(weight_decay)
            b_regularizer = regularizers.l2(weight_decay)
        else:
            W_regularizer = None
            b_regularizer = None
        x = Convolution2D(nb_filter, nb_row, nb_col,
                          subsample=subsample,
                          activation=activation,
                          border_mode=border_mode,
                          W_regularizer=W_regularizer,
                          b_regularizer=b_regularizer,
                          dim_ordering=dim_ordering)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        return x

    # Define image input layer

    if DIM_ORDERING == 'th':
        img_input = Input(shape=(3, 299, 299))
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        img_input = Input(shape=(299, 299, 3))
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Entry module

    x = conv2D_bn(
        img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid'
        )
    x = conv2D_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2D_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)

    x = conv2D_bn(x, 80, 1, 1, border_mode='valid')
    x = conv2D_bn(x, 192, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)

    # mixed: 35 x 35 x 256

    branch1x1 = conv2D_bn(x, 64, 1, 1)

    branch5x5 = conv2D_bn(x, 48, 1, 1)
    branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 32, 1, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # mixed_1: 35 x 35 x 288

    branch1x1 = conv2D_bn(x, 64, 1, 1)

    branch5x5 = conv2D_bn(x, 48, 1, 1)
    branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 32, 1, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # mixed2: 35 x 35 x 288

    branch1x1 = conv2D_bn(x, 64, 1, 1)

    branch5x5 = conv2D_bn(x, 48, 1, 1)
    branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 64, 1, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # mixed3: 17 x 17 x 768

    branch3x3 = conv2D_bn(
        x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(
        branch3x3dbl, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D(
        (3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # mixed4: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 128, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 128, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 1, 7)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # mixed5: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 160, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 160, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl,
              branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed5: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 160, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 160, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # mixed6: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 160, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 160, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # mixed7: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 192, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 160, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # Auxiliary head

    aux_logits = AveragePooling2D(
        (5, 5), strides=(3, 3), dim_ordering=DIM_ORDERING)(x)
    aux_logits = conv2D_bn(aux_logits, 128, 1, 1)
    aux_logits = conv2D_bn(aux_logits, 728, 5, 5, border_mode='valid')
    aux_logits = Flatten()(aux_logits)
    aux_preds = Dense(NB_CLASS, activation='softmax')(aux_logits)

    # mixed8: 8 x 8 x 1280

    branch3x3 = conv2D_bn(x, 192, 1, 1)
    branch3x3 = conv2D_bn(
        branch3x3, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2D_bn(x, 192, 1, 1)
    branch7x7x3 = conv2D_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2D_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2D_bn(
        branch7x7x3, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D(
        (3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = merge([branch3x3, branch7x7x3, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # mixed9: 8 x 8 x 2048

    branch1x1 = conv2D_bn(x, 320, 1, 1)

    branch3x3 = conv2D_bn(x, 384, 1, 1)
    branch3x3_1 = conv2D_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2D_bn(branch3x3, 384, 3, 1)
    branch3x3 = merge([branch3x3_1, branch3x3_2],
                      mode='concat', concat_axis=CONCAT_AXIS)

    branch3x3dbl = conv2D_bn(x, 448, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2D_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2D_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                         mode='concat', concat_axis=CONCAT_AXIS)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch3x3, branch3x3dbl,
              branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed10: 8 x 8 x 2048

    branch1x1 = conv2D_bn(x, 320, 1, 1)

    branch3x3 = conv2D_bn(x, 384, 1, 1)
    branch3x3_1 = conv2D_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2D_bn(branch3x3, 384, 3, 1)
    branch3x3 = merge([branch3x3_1, branch3x3_2],
                      mode='concat', concat_axis=CONCAT_AXIS)

    branch3x3dbl = conv2D_bn(x, 448, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2D_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2D_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                         mode='concat', concat_axis=CONCAT_AXIS)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), border_mode='same',
        dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=CONCAT_AXIS)

    # Final pooling and prediction

    x = AveragePooling2D(
        (8, 8), strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    preds = Dense(NB_CLASS, activation='softmax')(x)

    # Define model

    model = Model(input=img_input, output=[preds, aux_preds])

    # Keras accepts shape as
    #   (output_channels, input_channels, height, width)
    # For example, (32, 3, 3, 3) means there are 32
    # filters, with 3 channels, 3pixel height 3pixel width,
    # and bias is stored at param_1 and its shape (32,),
    # one for each filter.
    # https://github.com/fchollet/keras/issues/91
    # TensorFlow weight matrix is of order:
    #   (height, width, input_channels, output_channels)
    # https://goo.gl/eRKmJv

    with h5py.File('/home/tammy/www/model_zoo/inception-v3-hdf5-20160301/conv.h5', 'r') as hf:
        weights = hf['weights'][()].transpose((3, 2, 0, 1))
        model.layers[1].set_weights([weights, np.zeros((weights.shape[0],))])

    if weights_path:
        model.load_weights(weights_path)

    return model


def VGG_19(weights_path=None):
    '''VGG-19 model, source from https://goo.gl/rvcNDw'''

    layers = OrderedDict([
        ('conv1', [
            ZeroPadding2D((1, 1), input_shape=(3, 224, 224)),
            Convolution2D(64, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(64, 3, 3, activation='relu'),
            MaxPooling2D((2, 2),  strides=(2, 2))
        ]),
        ('conv2', [
            ZeroPadding2D((1, 1)),
            Convolution2D(128, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(128, 3, 3, activation='relu'),
            MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        ('conv3', [
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        ('conv4', [
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        ('conv5', [
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        ('fc', [
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
        ]),
        ('classify', [
            Dense(1000, activation='softmax')
        ])
    ])
    model = Sequential()
    for stack in layers:
        for l in layers[stack]:
            model.add(l)

    if weights_path:
        model.load_weights(weights_path)

    return model


def VGG_16(weights_path=None, lastFC=True):
    '''VGG-16 model, source from https://goo.gl/qqM88H'''

    layers = OrderedDict([
        ('conv1', [
            ZeroPadding2D((1, 1), input_shape=(3, 224, 224)),
            Convolution2D(64, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(64, 3, 3, activation='relu'),
            MaxPooling2D((2, 2),  strides=(2, 2))
        ]),
        ('conv2', [
            ZeroPadding2D((1, 1)),
            Convolution2D(128, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(128, 3, 3, activation='relu'),
            MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        ('conv3', [
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(256, 3, 3, activation='relu'),
            MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        ('conv4', [
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        ('conv5', [
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            ZeroPadding2D((1, 1)),
            Convolution2D(512, 3, 3, activation='relu'),
            MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        ('fc', [
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
        ]),
        ('classify', [
            Dense(1000, activation='softmax')
        ])
    ])
    model = Sequential()
    for stack in layers:
        if stack == 'classify' and not lastFC:
            print('[DPModels] Skip the last FC layer')
            continue
        for l in layers[stack]:
            model.add(l)

    if weights_path:
        load_weights(model, weights_path, lastFC=lastFC)

    return model


def load_weights(model, weights_path, lastFC=True):
    """ Load model weights

    @param model: the model
    @param weight_path: path of the weight file

    Arguments:

    lastFC -- True to load weights for the last FC layer (default: True)

    """
    if lastFC:
        model.load_weights(weights_path)
        return
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        ws = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(ws)
    f.close()
    return


def Simple(cats, img_row=224, img_col=224, conv_size=3,
           colors=3, weights_path=None, filter_size=32):
    '''Simple conv model to the best MNIST result.
       The input shape should be (n_imgs, colors, img_row, img_col).

    @param cats: total number of final categories

    Arguments:
    img_row -- number of rows of the image
    img_col -- number of cols of the image
    conv_size -- size of the conv window
    colors  -- colors of the image (3 for RGB)
    weights_path -- used for loading the existing model
    filter_size -- number of the filters of the first conv layer

    '''
    layers = OrderedDict([
        ('conv1', [
            Convolution2D(filter_size, conv_size, conv_size,
                          activation='relu', border_mode='valid',
                          input_shape=(colors, img_row, img_col)),
            Convolution2D(filter_size*2, conv_size, conv_size,
                          activation='relu'),
            MaxPooling2D((2, 2),  strides=(1, 1)),
            Dropout(0.25)
        ]),
        ('fc', [
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
        ]),
        ('classify', [
            Dense(cats, activation='softmax')
        ])
    ])
    model = Sequential()
    for stack in layers:
        for l in layers[stack]:
            model.add(l)

    if weights_path:
        model.load_weights(weights_path)

    return model
