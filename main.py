import cv2
from models import PLDTGAN
from fileLoader import getFiles , parse_Filenames , get_disassociated , loadFiles , processImages
import numpy as np
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
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


    
    Mod = PLDTGAN((64,64,3) , 64 , 2)
    #plot_model(Mod.Discrm , show_shapes=True , to_file = 'model.png')
    print(len(X))
    print(len(X_train[0]))
    print(len(X_train[-1]))
    #Mod.train(X_train , y_train , Targets)
    


   
    
    #resizeImages('lookbook/data/' , 'lookbook/resized/' , (64,64))



if __name__ == "__main__":
    main()