import googlenet_model as gmodel
import vgg_model as vmodel
import resnet_model as rmodel
from util.facial_expression_constants import NetType
from util import utils

class FaceModel(object):
    def __init__(self, net_type, input_shape, drop_out, drop_out_ratio, batch_normalization, activation, num_class=7):
        assert net_type
        assert input_shape
        assert drop_out
        assert drop_out_ratio
        assert batch_normalization
        assert activation
        assert num_class

        self.net_type = net_type
        self.input_shape = input_shape
        self.drop_out = drop_out
        self.drop_out_ratio = drop_out_ratio
        self.batch_normalization = batch_normalization
        self.activation = activation
        self.num_class = num_class
        
    # get model instance
    def get_model_instance(self):
        return model_factory(net_type=self.net_type, 
                            input_shape=self.input_shape,
                            drop_out=self.drop_out,
                            drop_out_ratio=self.drop_out_ratio
                            batch_normalization=self.batch_normalization,
                            activation=self.activation)
        

    # Model Factory generate the model
    def model_factory(self, net_type, input_shape, drop_out, drop_out_ratio, batch_normalization, activation, num_class):
        # Set Res110 as the default model
        model = rmodel.resnet_v2(input_shape=input_shape,
                                num_classes=num_class,
                                drop_out=drop_out,
                                drop_out_ratio=drop_out_ratio
                                bn=batch_normalization,
                                acti=activiation)
        
        if self.net_type == NetType.VGG:
            print('Choose the VGG model')
            model = vmodel.vgg(input_shape=input_shape,
                                num_classes=num_class,
                                drop_out=drop_out,
                                drop_out_ratio=drop_out_ratio,
                                batch_normalization=batch_normalization,
                                activation=activiation)
        elif net_type == NetType.GOOGLENET:
            print('Choose the GoogleNet model')
            model = gmodel.google_net(input_shape=input_shape, 
                                    num_classes=num_class, 
                                    drop_out=drop_out, 
                                    drop_out_ratio=drop_out_ratio, 
                                    batch_normalization=batch_normalization, 
                                    activation=activation):

        return model
