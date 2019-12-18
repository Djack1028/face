from util import utils
from util.facial_expression_constants import NetType,FacialExpressionType,OptimizerName
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import os
from model import FaceModel
import keras.metrics as metric

#********************************************************************************************************************************************
# Optimizer methdology
#--------------------------------------------------------------------------------------------------------------------------------------
# Optimizer Engine   | Data Augementation            | Dropout              | Batch Normalization               | Activation           |   
#--------------------------------------------------------------------------------------------------------------------------------------
# set opm_name       | set opm_data_augementation    | set opm_drop_out     | set opm_batch_normalization       | set Activation       |
# opm_name = Adam    | opm_data_augementation = True | opm_drop_out = True  | opm_batch_normalization = True    | activiation = 'relu' |
# opm_name = RMSprop |                               | drop_out_ratio = 0.4 |                                   |                      |
#---------------------------------------------------------------------------------------------------------------------------------------
# 
# 3rd type of modle for the analysis/comparison
# 1: VGG
# 2: GoogleNet
# 3: ResNet110
#
# Evaluation indexes
# 1: Loss value
# 2: Top 5 accuracy
# 3: In total accuracy
#*********************************************************************************************************************************************

################Hyper Parameters################
# modle selection
net_type = NetType.RESNET

# Optimizer Method
opm_data_augementation = True
opm_drop_out = True
opm_batch_normalization = True

# name for Optimizer
opm_name = OptimizerName.Adam

# Learning rate epoch
lr_epoch = 0

# Training parameters
batch_size = 32
epochs = 200

# Activation
activiation = 'relu'

# Facial Expression class num
num_facial_expression = 7

# Drop out ratio
drop_out_ratio = 0.4

################Prepare the Training Model##################
# initiate 
lr = utils.lr_schedule(lr_epoch)
opm = utils.get_optimizer(opm_name, lr, decay=1e-6)

#**********************Need replace the data source*********
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
#**********************Need replace the data source*********

# prepare the traning model
faceModel = FaceModel(net_type=net_type, 
                    input_shape=input_shape, 
                    drop_out=opm_drop_out, 
                    drop_out_ratio=drop_out_ratio
                    batch_normalization=opm_batch_normalization,
                    activiation=activiation,
                    num_class=num_facial_expression)

modle = faceModel.get_model_instance()
model.compile(loss='categorical_crossentropy',
              optimizer=opm,
              metrics=['accuracy',metric.top_k_categorical_accuracy])
model.summary()
print(model_type)

# Prepare model saving directory.
model_saved_path = utils.get_saved_model_path(net_type)
print(model_saved_path)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=model_saved_path,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(utils.lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

################Start with the Model Training#################
if not opm_data_augementation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4,
                        verbose=1,
                        callbacks=callbacks)

# Score trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Top 5 accuracy:', scores[2])
