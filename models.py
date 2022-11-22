from __future__ import division
from keras.layers import Reshape, TimeDistributed, Flatten, RepeatVector, Permute, Multiply, Add, UpSampling2D
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from numpy import concatenate
from tensorflow.python.keras.layers import Concatenate, BatchNormalization, MaxPooling3D

from dcn_vgg import dcn_vgg
import keras.backend as K

def schedule_vgg(epoch):
    lr = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5,
          1e-5, 1e-6, 1e-6, 1e-7, 1e-7]
    return lr[epoch]

# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    # max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1)), shape_r_out, axis=1)), shape_c_out, axis=2)
    max_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_pred /= max_y_pred

    max_y_true = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')

    sum_y_true = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    sum_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())
    return 10 * K.sum(y_bool * y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()))


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_pred /= max_y_pred

    # max_y_true = K.expand_dims(K.repeat_elements(
    #     K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), shape_r_out, axis=2)),
    #     shape_c_out, axis=3))
    max_y_true = K.max(y_true, axis=[2, 3, 4])
    y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')

    sum_y_true = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_true, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    sum_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = y_pred._shape_as_list()[2] * y_pred._shape_as_list()[3]
    sum_prod = K.sum(y_true * y_pred, axis=[2, 3, 4])
    sum_x = K.sum(y_true, axis=[2, 3, 4])
    sum_y = K.sum(y_pred, axis=[2, 3, 4])
    sum_x_square = K.sum(K.square(y_true), axis=[2, 3, 4])+ K.epsilon()
    sum_y_square = K.sum(K.square(y_pred), axis=[2, 3, 4])+ K.epsilon()

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return K.sum(y_bool*(-2 * num/den))#


# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    max_y_pred = K.expand_dims(K.repeat_elements(
        K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
        y_pred.shape[3], axis=3))
    y_pred /= max_y_pred
    # y_pred_flatten = K.batch_flatten(y_pred)

    # max_y_true = K.expand_dims(K.repeat_elements(
    #     K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), shape_r_out, axis=2)),
    #     shape_c_out, axis=3))
    max_y_true = K.max(y_true, axis=[2, 3, 4])
    y_bool = K.cast(K.greater(max_y_true, 0.1), 'float32')

    y_mean = K.mean(y_pred, axis=[2, 3, 4])
    y_mean = K.expand_dims(K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(y_mean),
                           y_pred.shape[2], axis=2)), y_pred.shape[3], axis=3))

    y_std = K.std(y_pred, axis=[2, 3, 4])
    y_std = K.expand_dims(K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(y_std),
                           y_pred.shape[2], axis=2)), y_pred.shape[3], axis=3))

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -K.sum(y_bool*((K.sum(y_true * y_pred, axis=[2, 3, 4])) / (K.sum(y_true, axis=[2, 3, 4]))))

# Let's use pure LSTM with VGG - and preprocessing layer. The preprocessing layer shall have merged effect.
# the encoder is VGG with LSTM
# the decoder is conv2D
# No Conv2D in the encoder part.

# Model 1  - VGG  + Two consecuative ConvLSTM
def acl_vgg(data, stateful):
    dcn = dcn_vgg()
    # print(data[0].shape, "Shape of orignal data") # DCN downsamples the image by 8 X 8

    outs = TimeDistributed(dcn)(data)
    # outs = dcn(data) #DCN downsamples the image by 8 X 8

    print(outs.shape, " After VGG")
    layer = ConvLSTM2D(filters=256, kernel_size=(3, 3)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation='tanh'
                       , padding='same', return_sequences=True, stateful=stateful)(outs) # 32 x40x 256

    print(layer.shape, " After First ConvLSTM" )

    layer = ConvLSTM2D(filters=256, kernel_size=(3, 3)
                                 , data_format='channels_last'
                                 , padding='same', return_sequences=True, stateful=stateful)(layer)

    print(layer.shape, " After Second LSTM")

    outs = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(layer)
    print(outs.shape, " After Conv1x1 ")

    attention = TimeDistributed(UpSampling2D(2))(outs)
    outs = TimeDistributed(UpSampling2D(4))(outs)

    print(outs.shape, attention.shape)
    return [outs, outs, outs, attention, attention, attention]

## Model 2 - ?
# def acl_vgg(data, stateful):
#     dcn = dcn_vgg()
#     # print(data[0].shape, "Shape of orignal data") # DCN downsamples the image by 8 X 8
#
#     outs = TimeDistributed(dcn)(data)
#     # outs = dcn(data) #DCN downsamples the image by 8 X 8
#
#     print(outs.shape, " After VGG")
#     layer = ConvLSTM2D(filters=256, kernel_size=(3, 3)
#                        , data_format='channels_last'
#                        , recurrent_activation='hard_sigmoid'
#                        , activation='tanh'
#                        , padding='same', return_sequences=True, stateful=stateful)(outs) # 32 x40x 256
#
#     print(layer.shape, " After First ConvLSTM" )
#
#     layer = ConvLSTM2D(filters=256, kernel_size=(3, 3)
#                                  , data_format='channels_last'
#                                  , padding='same', return_sequences=True, stateful=stateful)(layer)
#
#     print(layer.shape, " After Second LSTM" )
#
#     outs = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(layer)
#
#     attention = TimeDistributed(UpSampling2D(2))(outs)
#     outs = TimeDistributed(UpSampling2D(4))(outs)
#
#     print(outs.shape, attention.shape)
#     return [outs, outs, outs, attention, attention, attention]






































# Model A  - Orignal Model
# After VGG  (None, None, 32, 40, 512)
# Attention Size  (None, None, 32, 40, 1)
# Before LSTM  (None, None, 32, 40, 512)
# After First LSTM  (None, None, 32, 40, 256)
# Outs Shape  (None, None, 32, 40, 1)
# (None, None, 128, 160, 1) (None, None, 64, 80, 1)

# def acl_vgg(data, stateful):
#     dcn = dcn_vgg()
#     # print(data[0].shape, "Shape of orignal data") # DCN downsamples the image by 8 X 8
#     # outs = TimeDistributed(dcn)(data[0])
#     outs = TimeDistributed(dcn)(data) #DCN downsamples the image by 8 X 8
#     print("After VGG ", outs.shape)
#     # downsampled = TimeDistributed(Conv2D(512, (3, 3), padding='same', activation='relu'))(data[1])
#     # downsampled = TimeDistributed(MaxPooling2D((2, 2), strides=(8, 8), padding='same'))(downsampled)
#     # outs = Add()([outs, downsampled])
#
#
#     attention = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(outs)
#     attention = TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(attention)
#     attention = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(attention)
#     attention = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))(attention)
#     attention = TimeDistributed(Conv2D(64, (1, 1), padding='same', activation='relu'))(attention)
#     attention = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(attention)
#
#     attention = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(attention)
#     attention = TimeDistributed(UpSampling2D(4))(attention)
#
#     print("Attention Size ", attention.shape) #32 x 40 x 1
#
#     # attention = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))(outs)
#     # attention = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(attention)
#     # attention = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(attention)
#
#     f_attention = TimeDistributed(Flatten())(attention)
#     f_attention = TimeDistributed(RepeatVector(512))(f_attention)
#     f_attention = TimeDistributed(Permute((2, 1)))(f_attention)
#     f_attention = TimeDistributed(Reshape((32, 40, 512)))(f_attention)#30
#
#     # Residual ? <--
#     m_outs = Multiply()([outs, f_attention])
#     outs = Add()([outs, m_outs])
#     print("Before LSTM ", outs.shape)
#     outs = (ConvLSTM2D(filters=256, kernel_size=(3, 3),
#                        padding='same', return_sequences=True, stateful=stateful, dropout=0.4))(outs)
#
#     print("After First LSTM ", outs.shape)
#
#     outs = TimeDistributed(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))(outs)
#
#     print("Outs Shape ", outs.shape)
#     outs = TimeDistributed(UpSampling2D(4))(outs)
#     attention = TimeDistributed(UpSampling2D(2))(attention)
#
#     print(outs.shape, attention.shape)
#     return [outs, outs, outs, attention, attention, attention]