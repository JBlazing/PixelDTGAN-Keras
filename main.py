import cv2
from models import PLDTGAN
from fileLoader import getFiles , parse_Filenames , get_disassociated , loadFiles , processImages
import numpy as np
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.python import debug as tf_debug
import tensorflow as tf

import gc
def main():
    
    files = getFiles('lookbook/resized/')
    
    X,Targets,Y_idxs = parse_Filenames(files)
    
    X = np.array(X)
    

    dis = get_disassociated(Y_idxs , len(Targets))
    Targets = np.array(Targets)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X , np.array([ (x,y) for x, y in zip(Y_idxs , dis )]) , train_size=.9 )

    splits = int(X_train.shape[0] / 128)
    
    X_train = np.array_split(X_train , splits)
    y_train = np.array_split(y_train, splits)

    
    

    tf.enable_eager_execution()
    
    Mod = PLDTGAN((64,64,3) , 64 , 25)
    #plot_model(Mod.GAN , show_shapes=True , to_file = 'model.png')


    Mod.train(X_train , y_train , Targets)
    



    
    #resizeImages('lookbook/data/' , 'lookbook/resized/' , (64,64))



if __name__ == "__main__":
    main()