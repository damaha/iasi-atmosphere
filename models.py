from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.layers import Conv2D, Conv3D, Activation, Input, Concatenate
from keras.layers import Reshape, AveragePooling2D, MaxPooling2D, Dropout
from keras.layers import UpSampling2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer

def simple_model_1(insize, output_modes, n_comp=128, f_spec=128):
    inputs = Input(shape=insize)

    x = Reshape(insize[:2]+(1,)+insize[2:])(inputs)
    x = Conv3D(n_comp, kernel_size=(1, 1, f_spec), padding='same', activation='sigmoid')(x)
    x = Reshape(insize[:2]+(n_comp,))(x)

    x = Conv2D(12, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(12, (3, 3), padding='same', activation='relu')(x)

    # x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    # x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    predictions = Conv2D(output_modes, (1, 1), padding='same', activation=None)(x)

    model = Model(inputs=inputs, outputs=predictions)

    return(model)

def simple_model_2(insize, output_modes, n_comp=128, f_spec=128):
    inputs = Input(shape=insize)

    x = Conv2D(f_spec, (1, 1), padding='same', activation='relu')(inputs)

    x = Conv2D(12, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(12, (3, 3), padding='same', activation='relu')(x)

    x = Dropout(0.25)(x)

    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x = Dropout(0.25)(x)
    
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    predictions = Conv2D(output_modes, (1, 1), padding='same', activation=None)(x)

    model = Model(inputs=inputs, outputs=predictions)

    return(model)

def simple_model_3(insize, output_modes, n_comp=128, f_spec=128):
    inputs = Input(shape=insize)

    x = Conv2D(n_comp, (1, 1), padding='same', activation='relu')(inputs)

    x = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    
    x = Dropout(0.25)(x)

    x = Conv2D(128, (1, 1), padding='same', activation='relu')(x)

    predictions = Conv2D(output_modes, (1, 1), padding='same', activation=None)(x)

    model = Model(inputs=inputs, outputs=predictions)

    return(model)

def ASPP_model(insize, n_comp=128, f_spec=128):
    inputs = Input(shape=insize)

    x = Reshape(insize+[1])(inputs)
    x = Conv3D(n_comp, kernel_size=(1, 1, f_spec), padding='same', activation='sigmoid')(x)
    x = Reshape(insize[:2]+[n_comp])(x)

    x = Conv2D(12, (3, 3), padding='same', activation='relu')(x)
    x_ = Conv2D(12, (3, 3), padding='same', activation='relu')(x)

    x = BatchNormalization()(x_)
    x = Dropout(0.25)(x)

    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(18, (3, 3), padding='same', activation='relu')(x)

    x1 = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding="same")(x)
    x2 = AveragePooling2D(pool_size=(4,4), strides=(1,1), padding="same")(x)
    x3 = AveragePooling2D(pool_size=(8,8), strides=(1,1), padding="same")(x)
    x4 = AveragePooling2D(pool_size=(16,16), strides=(1,1), padding="same")(x)

    x1 = Conv2D(12, (3, 3), padding='same',
                  dilation_rate=(2, 2), activation='relu')(x1)    
    x2 = Conv2D(12, (3, 3), padding='same',
                  dilation_rate=(4, 4), activation='relu')(x2)
    x3 = Conv2D(12, (3, 3), padding='same',
                  dilation_rate=(8, 8), activation='relu')(x3)
    x4 = Conv2D(12, (3, 3), padding='same',
                  dilation_rate=(16, 16), activation='relu')(x4)

    x = Concatenate()([x_, x1, x2, x3, x4])
    predictions = Conv2D(1, (1, 1), padding='same', activation=None)(x)

    model = Model(inputs=inputs, outputs=predictions)

    return(model)