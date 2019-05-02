import cv2
from models import PLDTGAN
from fileLoader import getFiles , parse_Filenames , get_disassociated , loadFiles , processImages
import numpy as np

def main():
    
    files = getFiles('lookbook/resized/')
    
    X,Targets,Y_idxs = parse_Filenames(files)

    X, Targets = loadFiles(X,Targets)

    X , Targets = processImages(X , Targets)
    


    dis = get_disassociated(Y_idxs , len(Targets))
    
    Y_D_Images = [ (Targets[idx], Targets[d]) for idx , d in zip(Y_idxs , dis) ]
    
    Mod = PLDTGAN(X[0].shape , 64 , 64)

    Mod.train(X , Y_D_Images)
    
    

    '''
    dis = get_disassociated(Y , Y_idxs)

    
    cv2.imread('lookbook/resized/PID008105_CLEAN1_IID070296.jpg')

    X , Y = loadFiles(X , Y)
    print(Y)
    #a = cv2.imread(X[0])
    x = input("Size:")

    
    input_shape = (64,64,3)

    
    
    
    plot_model(Mod.GAN , show_shapes=True , to_file="model.png" )
    
    '''

   
    
    #resizeImages('lookbook/data/' , 'lookbook/resized/' , (64,64))



if __name__ == "__main__":
    main()