import googlenet_model as gmodel
import vgg_model as vmodel
import resnet_model as rmodel
from util.facial_expression_constants import NetType
from util import utils

class FaceModel(object):
    def __init__(self, net_type, input_shape, drop_out, batch_normalization, activation):
        assert net_type
        assert input_shape
        assert drop_out
        assert batch_normalization
        assert activation

        self.net_type = net_type
        self.input_shape = input_shape
        self.drop_out = drop_out
        self.batch_normalization = batch_normalization
        self.activation = activation
        
    # get model instance
    def get_model_instance(self):
        return model_factory(net_type=self.net_type, 
                            input_shape=self.input_shape,
                            drop_out=self.drop_out，
                            batch_normalization=self.batch_normalization,
                            activation=self.activation)
        

    # Model Factory generate the model
    def model_factory(self, net_type, input_shape, drop_out, batch_normalization, activation):
        # Set Res110 as the default model
        model = rmodel.resnet_v2(input_shape=input_shape,
                                num_classes=7,
                                drop_out=drop_out,
                                bn=batch_normalization,
                                acti=activiation)
        
        if self.net_type == NetType.VGG:
            model = vmodel.vgg(input_shape=input_shape,
                                num_classes=7,
                                drop_out=drop_out,
                                bn=batch_normalization,
                                acti=activiation)
        elif net_type == NetType.GOOGLENET:
            #mdoel = gmodel.get_model(input_shape)
            pass

        return model
