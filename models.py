
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *

def createLayers(input , outputSize , leaky=True , batch=True):
        l = Conv2D(outputSize ,  kernel_size = kernel_size , strides=strides)(inLayer)
        if Leaky:
            l = LeakyReLU(.2)(l)
        if batch:
            l = BatchNormalization()(l)
        return l

def createGen(input_shape , filters):

    def createInGenLayer(inLayer , outputSize , norm=True , kernel_size=(4,4) , strides=(2,2)):
        l = Conv2D(outputSize , kernel_size = kernel_size , strides=strides)(inLayer)
        l = LeakyReLU(.2)(l)
        if norm:
            l = BatchNormalization()(l)
        return l
    def createOutGenLayer(inLayer , outputSize , activation='relu' , norm=True , kernel_size=(4,4) , strides=(2,2)):
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
    GenModel.compile(loss='mean_squared_error', optimizer='sgd')

    return GenModel

def createDisc(input_shape , filters ):



    in_layer = Input(shape=input_shape , name="Discrm Input")
    L1 = createLayers(in_layer , filters)
    L2 = createLayers(L1 , filters * 2)
    L3 = createLayers(L2 , filters * 4)
    L4 = createLayers(L3 , filters * 8)
    L5 = createLayers(L4 , 1 , leaky=False , batch=False)
    L6 = Activation('sigmoid')(L5)
    '''
    DiscModel = Model(inputs=[in_layer] , outputs=[L6])
    DiscModel.compile()
    '''
    return L6

def createAssociated(inputs , filters):

    InCat = keras.Concatenate(name="Cat_Input")(inputs)          
    L1 = createLayers(InCat , filters)
    L2 = createLayers(L1 , filters * 2)
    L3 = createLayers(L2 , filters * 4)
    L4 = createLayers(L3 , filters * 8)
    L5 = createLayers(L4 , 1 , leaky=False , batch=False)
    L6 = Activation('sigmoid')(L5)
    '''
    AssocModel = Model()
    '''
    return L6