import enum

NetType = enum.Enum('NetType',('VGG','RESNET','GOOGLENET'))
FacialExpressionType = enum.Enum('FacialExpressionType',('ANGER','HAPPY'))
OptimizerName = enum.Enum('OptimizerName', ('Adam','RMSprop'))
