import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras import Model

#functional approach : function that returns a model
def functional_model():
    #for MNIST dataset
    my_inputs = Input(shape = (28,28,1))
    x = Conv2D(32,(3,3),activation = 'relu')(my_inputs)
    x = Conv2D(64,(3,3),activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3),activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64,activation = 'relu')(x)
    x = Dense(10,activation = 'softmax')(x) #10 represents no of output classes

    model = Model(inputs = my_inputs,outputs = x)
    return model

#inheriting from the class
class tf_class_model(Model):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2D(32,(3,3),activation = 'relu')
        self.conv2 = Conv2D(64,(3,3),activation = 'relu')
        self.mpool1 = MaxPool2D()
        self.bnorm1 = BatchNormalization()

        self.conv3 = Conv2D(138,(3,3),activation = 'relu')
        self.mpool2 = MaxPool2D()
        self.bnorm2 = BatchNormalization()

        self.gavg1 = GlobalAvgPool2D()
        self.dense1 = Dense(64,activation = 'relu')
        self.dense2 = Dense(10,activation = 'softmax')

    def call(self,my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

def street_sign_model(no_of_classes):
    my_input = Input(shape = (60,60,3))

    x = Conv2D(32,(3,3),activation = 'relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64,(3,3),activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3),activation = 'relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128,activation = 'relu')(x)
    x = Dense(no_of_classes,activation = 'softmax')(x)

    return Model(inputs = my_input,outputs = x)

if __name__ == '__main__':
    model = street_sign_model(42)
    model.summary()