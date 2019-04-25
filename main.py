import cv2
from models import createGen 
import fileLoader
from customLayer import ImageLayer
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model



def main():
    '''
    files = fileLoader.getFiles('lookbook/data/')
    GenModel , DiscModel , AModel = models.createModels((64,64,2))
    '''
    input_shape = (64,64,3)

    
    GenLayers = createGen(input_shape , 64)
    
    

    
    
    


if __name__ == "__main__":
    main()