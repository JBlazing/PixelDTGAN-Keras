import cv2
from models import PLDTGAN
from tensorflow.keras.utils import plot_model



def main():
    
    '''
    dir(K)
    
    
    files = fileLoader.getFiles('lookbook/data/')
    GenModel , DiscModel , AModel = models.createModels((64,64,2))
    '''
    input_shape = (64,64,3)

    
    Mod = PLDTGAN(input_shape , 64 , 64)
    
    plot_model(Mod.Assoc , show_shapes=True , to_file="model.png" )
    

    


if __name__ == "__main__":
    main()