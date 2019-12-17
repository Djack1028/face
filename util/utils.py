from facial_expression_constants import NetType,OptimizerName
from keras.optimizers import Adam, RMSprop

def get_model_name(net_type):
    assert net_type
    if net_type == NetType.VGG:
        return 'VGG'
    elif net_type == NetType.GOOGLENET:
        return 'GOOGLENET'
    else:
        return'RESNET'

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Generate Optimizer automatically using for the optimizer analysis
def get_optimizer(opm_name=OptimizerName.Adam, lr=0.9,decay=0):
    opm = Adam(learning_rate=lr,decay=decay)

    if opm_name == OptimizerName.RMSprop:
        opm = RMSprop(learning_rate=lr, decay=decay) 
   
    return opm

# Generate path used to save model
def get_saved_model_path(net_type):
    assert net_type
    modle_name = get_model_name(net_type)

    save_dir = os.path.join(os.path.abspath('..'), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    print('The saved modle path is %s' % filepath)

    return filepath
