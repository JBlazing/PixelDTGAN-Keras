import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import *
from keras.utils import *



def createModels(input_shape):


    GenModel = Sequential()


    GenModel.add(Conv2D(96, kernel_size=(4, 4), strides=(2, 2),input_shape=input_shape))
    GenModel.add(LeakyReLU(.2))
    GenModel.add(Conv2D(96 * 2, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(BatchNormalization())
    GenModel.add(LeakyReLU(.2))
    GenModel.add(Conv2D(96 * 4, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(BatchNormalization())
    GenModel.add(LeakyReLU(2))
    GenModel.add(Conv2D(96 * 8, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(BatchNormalization())
    GenModel.add(LeakyReLU(.2))

    GenModel.add(Conv2D(96 * 4, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(BatchNormalization())
    GenModel.add(Activation('relu'))
    GenModel.add(Conv2D(96 * 2, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(BatchNormalization())
    GenModel.add(Activation('relu'))
    GenModel.add(Conv2D(96 * 4, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(BatchNormalization())
    GenModel.add(Activation('relu'))
    GenModel.add(Conv2D(96 * 2, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(BatchNormalization())
    GenModel.add(Activation('relu'))
    GenModel.add(Conv2D(96, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(BatchNormalization())
    GenModel.add(Activation('relu'))
    GenModel.add(Conv2D(3, kernel_size=(4, 4), strides=(2, 2)))
    GenModel.add(Activation('tanh'))


    DiscModel = Sequential()

    DiscModel.add(Conv2D(96, kernel_size=(4, 4), strides=(2, 2),input_shape=input_shape))
    DiscModel.add(LeakyReLU(.2))
    DiscModel.add(Conv2D(96 * 2, kernel_size=(4, 4), strides=(2, 2)))
    DiscModel.add(BatchNormalization())
    DiscModel.add(LeakyReLU(.2))
    DiscModel.add(Conv2D(96 * 4, kernel_size=(4, 4), strides=(2, 2)))
    DiscModel.add(BatchNormalization())
    DiscModel.add(LeakyReLU(.2))
    DiscModel.add(Conv2D(96 * 8, kernel_size=(4, 4), strides=(2, 2)))
    DiscModel.add(BatchNormalization())
    DiscModel.add(LeakyReLU(.2))
    DiscModel.add(Conv2D(1, kernel_size=(4, 4)))
    DiscModel.add(Dense(1 , Activation='sigmoid'))

    AModel = Sequential()

    AModel.add(Conv2D(96 , kernel_size=(4,4) , strides = (2,2) , input_shape = (64,64,6)))
    AModel.add(LeakyReLU(.2))
    AModel.add(Conv2D(96 * 2 , kernel_size=(4,4) , strides = (2,2) ))
    AModel.add(BatchNormalization())
    AModel.add(LeakyReLU(.2))
    AModel.add(Conv2D(96 * 4 , kernel_size=(4,4) , strides = (2,2) ))
    AModel.add(BatchNormalization())
    AModel.add(LeakyReLU(.2))
    AModel.add(Conv2D(96 * 8 , kernel_size=(4,4) , strides = (2,2) ))
    AModel.add(BatchNormalization())
    AModel.add(LeakyReLU(.2))
    AModel.add(Conv2D(1 , kernel_size=(4,4)))
    AModel.add(Activation('sigmoid'))
    AModel.add(Dense(1))

    return GenModel , DiscModel  , AModel