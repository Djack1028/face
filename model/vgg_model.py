import keras
from keras.models import Model
from keras.layers import Activation, Conv2D, Dense, BatchNormalization
from keras.layers import Input, Flatten, MaxPooling2D

################用于正则化时权重降低的速度
# weight_decay = 0.0005
# nb_epoch=100
# batch_size=32

# VGG convolution block
# 1 and 2th layer have 2 convs and 1 pooling
# 3 -- 5 layer have 3 convs and 1 pooling
def vgg_conv_block(inputs, kernel_size=3, num_filters, strides=1, activation='relu',batch_normalization=True, num_layer):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs

    conv_times = 3
    if num_layer < 2:
        conv_times = 2
    
    for num_conv in range(conv_times):
        print('%dth executed convolution in %dth VGG layer' % num_conv+1, num_layer+1)
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)

    # Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    feature_map_dimension = 224/2**(num_layer+1)
    print('The feature maps in the %dth conv layer is %d*%d*%d' % num_layer+1, num_filters, feature_map_dimension, feature_map_dimension)

    return x

def vgg_fully_conn_block(inputs, out_put_size, activation='relu', drop_out, drop_out_ratio=0.4, batch_normalization=True):
    y = Dense(out_put_size,
                activation=activation,
                kernel_initializer='he_normal')(inputs)
    
    if batch_normalization:
        y = BatchNormalization()y
    
    
    # Add the Dropout
    if drop_out:
        y = Dropout(drop_out_ratio)(y)

    return y

#####################################################################
# Default 7 types of facial expression
# VGG 16 Weight layer
# ---------------------------------------------------
# |_____Layer_____|__Convovlution__|____Pooling____|____Output_____|
# |_______1_______|_______2________|_______1_______|__112*112*128__|
# |_______2_______|_______2________|_______1_______|___56*56*256___|
# |_______3_______|_______3________|_______1_______|___28*28*512___|
# |_______4_______|_______3________|_______1_______|___14*14*512___|
# |_______5_______|_______3________|_______1_______|____7*7*512____|
# |_______3_______|_______1________|_______1_______|___28*28*512___|
# 
# Fully conection layer in total 4
# in total 2 layer of 4096 fully connection
# 1 layer of 1000 fully connection
# 1 layer of 7 soft-max
#
# Only drop out in the Fully connection layer
####################################################################
def vgg(input_shape, num_classes=7, drop_out, drop_out_ratio=0.4,batch_normalization, activation):
    inputs = Input(shape=input_shape)
    x = inputs
    # initiate the filter
    num_filters = 64

    # Convolution layer
    # can add drop out later
    for num_layer in range(5):
        # double filters for each layer in previous 4 layer
        if num_layer < 4:
            num_filters *= 2
        x = vgg_conv_block(x, 
                    num_filters=num_filters,
                    activation=activation,
                    batch_normalization=batch_normalization,
                    num_layer=num_layer)
    
    
    # Fully connection layer
    y = Flatten()(x)
    # default set as 4096 
    out_put_size = 4096
    for num_connection in range(3):
        y = vgg_fully_conn_block(y, 
                                out_put_size, 
                                activation=activation,
                                batch_normalization=batch_normalization,
                                drop_out=drop_out,
                                drop_out_ratio=drop_out_ratio)
        
        if num_connection == 1:
            # out put set as 1000 in 3rd fully connection 
            out_put_size = 1000
    
    # Outputs
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
