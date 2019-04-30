import cv2
from models import PLDTGAN
from tensorflow.keras.utils import plot_model
from fileLoader import getFiles , parse_Filenames , get_disassociated , loadFiles


def main():
    
  
    files = getFiles('lookbook/resized/')
    
    X,Targets,Y_idxs = parse_Filenames(files)

    X, Targets = loadFiles(X,Targets)

    Y_image = [ Targets[idx] for idx in Y_idxs ]
    
    
    print(len(Y_idxs))
            
    input('a')

    '''
    dis = get_disassociated(Y , Y_idxs)

    
    cv2.imread('lookbook/resized/PID008105_CLEAN1_IID070296.jpg')

    X , Y = loadFiles(X , Y)
    print(Y)
    #a = cv2.imread(X[0])
    x = input("Size:")

    
    input_shape = (64,64,3)

    
    Mod = PLDTGAN(input_shape , 64 , 64)
    
    plot_model(Mod.GAN , show_shapes=True , to_file="model.png" )
    
    '''

   
    
    #resizeImages('lookbook/data/' , 'lookbook/resized/' , (64,64))



if __name__ == "__main__":
    main()