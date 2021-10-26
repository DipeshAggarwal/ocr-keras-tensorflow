from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

class ResNet:
    @staticmethod
    def residual_model(data, K, stride, chan_dim, reduce=False, reg=0.0001, bn_eps=2e-5, bn_mom=0.9):
        shortcut = data
        
        bn_1 = BatchNormalization(chan_dim, momentum=bn_mom, epsilon=bn_eps)(data)
        act_1 = Activation("relu")(bn_1)
        conv_1 = Conv2D(int(K*0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act_1)
        
        bn_2 = BatchNormalization(chan_dim, momentum=bn_mom, epsilon=bn_eps)(conv_1)
        act_2 = Activation("relu")(bn_2)
        conv_2 = Conv2D(int(K*0.25), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act_2)

        bn_3 = BatchNormalization(chan_dim, momentum=bn_mom, epsilon=bn_eps)(conv_2)
        act_3 = Activation("relu")(bn_2)
        conv_3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act_3)

        if reduce:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act_1)
            
        x = add([conv_3, shortcut])
        
        return x
    
    @staticmethod
    def build(width, height, classes, stages, filters, reg=0.0001, bn_eps=2e-5, bn_mom=0.9, dataset="cifar"):
        input_shape = (height, width, depth)
        chan_dim = -1
        
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
            
        inputs = Input(shape=input_shape)
        x = BatchNormalization(axis=chan_dim, momentum=bn_mon, epsilon=bn_eps)(inputs)
        x = Conv2D(filters[0], (3, 3), use_bias=False, kernel_regularizer=l2(reg))(x)
        
        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_model(x, filters[i+1], stride, chan_dim)
            
            for j in range(0, steps[i]-1):
               x = ResNet.residual_model(x, filters[i+1], (1, 1), chan_dim)
               
        x = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        
        model = Model(inputs, x, name="ResNet")
        
        return model
