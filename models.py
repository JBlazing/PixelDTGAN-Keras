import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import *
from keras.utils import *


'''
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

    opt = Adam(0.0002,0.5)

    Disc.compile(loss= 'mse' , optimizer = opt , metrics=['accuracy'])
    
    imgA = Input(shape=input_shape)
    imgB = Input(shape=input_shape)

    fakeA = GenModel(imgB)

    DiscModel.trainable = False
    valid = DiscModel([fakeA , imgB])

    combo = Model(inputs=([imgA , imgB]) , outputs=[valid , fake_A])

    return GenModel , DiscModel  , AModel

    '''

def createGen(input_shape , filters):

    def createInGenLayer(inLayer , outputSize , norm=True , kernel_size=(4,4) , strides=(2,2)):
        l = Conv2D(outputSize , kernel_size = kernel_size , strides=strides)(inLayer)
        l = LeakyReLU(.2)(l)
        if norm:
            l = BatchNormalization()(l)
        return l
    def createOutGenLayer(inLayer , outputSize , activation='relu' norm=True , kernel_size=(4,4) , strides=(2,2))
        l = Conv2DTranspose(outputSize , activation=activation , kernel_size=kernel_size , strides=strides)(inLayer)
        if norm:
            l = BatchNormalization()(l)
        return l
    
    in_layer = Input(shape=input_shape , name = "Input")

    G1 = createInGenLayer(in_layer , filters)
    G2 = createInGenLayer(G1 , filters * 2)
    G3 = createInGenLayer(G2, filters * 4)
    G4 = createInGenLayer(G3 , filters * 8)

    G5 = createOutGenLayer(G4 , filters * 4)
    G6 = createOutGenLayer(G5 , filters * 2)
    G7 = createOutGenLayer(G6 , filters)
    G8 = createOutGenLayer(G7 , 3 , activation='tanh')

    GenModel = Model(inputs=[in_layer] , outputs=[G8])
    GenModel.compile()

    return GenModel

def createDisc(input_shape , filters ):

    def createLayers(input , outputSize , leaky=True , batch=True):
        l = Conv2D(outputSize ,  kernel_size = kernel_size , strides=strides)(inLayer)
        if Leaky:
            l = LeakyReLU(.2)(l)
        if batch:
            l = BatchNormalization()(l)
        return l

    in_layer = Input(shape=input_shape , name="Discrm Input")
    L1 = createLayers(in_layer , filters)
    L2 = createLayers(L1 , filters * 2)
    L3 = createLayers(L2 , filters * 4)
    L4 = createLayers(L3 , filters * 8)
    L5 = createLayers(L4 , 1 , leaky=False , batch=False)
    L6 = Activation('sigmoid')(L5)

    DiscModel = Model(inputs=[in_layer] , outputs=[L6])
    DiscModel.compile()

    return DiscModel

