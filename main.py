import cv2
import models
import fileLoader
from customLayer import ImageLayer
from keras.layers import Input
from keras.models import Sequential, Model
from keras.callbacks import Callback

import numpy as np

def main():
    '''
    files = fileLoader.getFiles('lookbook/data/')
    GenModel , DiscModel , AModel = models.createModels((64,64,2))
    '''
    input_shape = (1,4)

    img_1 = Input(shape=input_shape , name = "Img_1")
    img_2 = Input(shape=input_shape , name = "Img_2")

    test = ImageLayer(input_shape)([img_1 , img_2])


    Mod = Model(inputs=[img_1 , img_2] , outputs=test)
    Mod.compile(loss='mean_squared_error', optimizer='sgd')

    inpu = [np.indices(input_shape) , np.indices(input_shape)]
    inpu[0][0][0][1] = 10
    print(inpu[1][0][0][1])
    
    
  

    output = Mod.predict({"Img_1" : inpu[0] , "Img_2" : inpu[1]})


    print(output)
    


if __name__ == "__main__":
    main()