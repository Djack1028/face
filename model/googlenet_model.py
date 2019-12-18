#-*- coding: UTF-8 -*-
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, Dense, Dropout,BatchNormalization
from keras.layers import Input, concatenate
from keras.models import Model
from keras import regularizers
import os

# Global Constants
WEIGHT_DECAY=0.0005
LRN2D_NORM=True
DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'

# Convolution with Normalization
def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY):
    #l2 normalization
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    #local response normalization
    if lrn2d_norm:        
        x=BatchNormalization()(x)

    return x


def inception_module(x,filters,concat_axis,padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=None):
    (branch1,branch2,branch3,branch4)=filters
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None
    #1x1
    pathway1=Conv2D(filters=branch1[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)

    #1x1->3x3
    pathway2=Conv2D(filters=branch2[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway2=Conv2D(filters=branch2[1],kernel_size=(3,3),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway2)

    #1x1->5x5
    pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)

    #3x3->1x1
    pathway4=MaxPooling2D(pool_size=(3,3),strides=1,padding=padding,data_format=DATA_FORMAT)(x)
    pathway4=Conv2D(filters=branch4[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway4)

    return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)


# input shape is 224*224*3
# only add the drop out in the fully connection layer
def google_net(input_shape, num_classes=7, drop_out=True, drop_out_ratio=0.4, batch_normalization=True, activation='relu'):
    #Data format:tensorflow,channels_last;theano,channels_last
    # need replace this input shape
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,224,224)
        inputs=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(224,224,3)
        inputs=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3
    else:
        raise Exception('Invalid Dim Ordering')
    
    # output is 64*56*56
    x=conv2D_lrn2d(inputs,64,(7,7),2,padding='same',lrn2d_norm=False,activation=activation)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    if batch_normalization
        x=BatchNormalization()(x)

    # output is 64*56*56
    x=conv2D_lrn2d(x,64,(1,1),1,padding='same',lrn2d_norm=False,activation=activation)
    if bacth_normalization:
        x = x=BatchNormalization()(x)
        
    # output is 192*28*28
    x=conv2D_lrn2d(x,192,(3,3),1,padding='same',lrn2d_norm=True,activation=activation)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    if bacth_normalization:
        x = x=BatchNormalization()(x)

    # output is 480*14*14
    x=inception_module(x,filters=[(64,),(96,128),(16,32),(32,)],concat_axis=CONCAT_AXIS,activation=activation) #3a
    x=inception_module(x,filters=[(128,),(128,192),(32,96),(64,)],concat_axis=CONCAT_AXIS,activation=activation) #3b
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    if bacth_normalization:
        x = x=BatchNormalization()(x)

    # output is 832*7*7
    x=inception_module(x,filters=[(192,),(96,208),(16,48),(64,)],concat_axis=CONCAT_AXIS,activation=activation) #4a
    x=inception_module(x,filters=[(160,),(112,224),(24,64),(64,)],concat_axis=CONCAT_AXIS,activation=activation) #4b
    x=inception_module(x,filters=[(128,),(128,256),(24,64),(64,)],concat_axis=CONCAT_AXIS,activation=activation) #4c
    x=inception_module(x,filters=[(112,),(144,288),(32,64),(64,)],concat_axis=CONCAT_AXIS,activation=activation) #4d
    x=inception_module(x,filters=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS,activation=activation) #4e
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    if bacth_normalization:
        x = x=BatchNormalization()(x)

    # output is 1024*1*1
    x=inception_module(x,filters=[(256,),(160,320),(32,128),(128,)],concat_axis=CONCAT_AXIS,activation=activation) #5a
    x=inception_module(x,filters=[(384,),(192,384),(48,128),(128,)],concat_axis=CONCAT_AXIS,activation=activation) #5b
    x=AveragePooling2D(pool_size=(7,7),strides=1,padding='valid',data_format=DATA_FORMAT)(x)
    if bacth_normalization:
        x = x=BatchNormalization()(x)

    # Fully connected layer
    x=Flatten()(x)
    if drop_out:
        x=Dropout(drop_out_ratio)(x)
    x=Dense(output_dim=num_classes,activation='linear')(x)
    x=Dense(output_dim=num_classes,activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=x)
    return model

    # return x,img_input,CONCAT_AXIS,INP_SHAPE,DATA_FORMAT


