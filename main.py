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
    
    X_train, X_test, y_train, y_test = train_test_split(X , np.array([ (x,y) for x, y in zip(Y_idxs , dis )]) )

    splits = int(X_train.shape[0] / 128)
    
    X_train = np.array_split(X_train , splits)
    y_train = np.array_split(y_train, splits)


    
    Mod = PLDTGAN((64,64,3) , 64 , 1)
    #plot_model(Mod.Discrm , show_shapes=True , to_file = 'model.png')
    Mod.train(X_train , y_train , Targets)
    
    #print(X[0])

    #for i, x in enumerate(I_Targets):
    #    I_Targets[i] = cv2.normalize(x, I_Targets[i], -1,  1 , cv2.NORM_MINMAX , cv2.CV_32FC3)
    #    print(I_Targets[i])
    #print(I_Targets)
    #I_X_Train = loadFiles(X_train)
    
    #print(I_Targets)

    #rain_Targets = np.stack([ I_Targets[y[0]] for y in y_train])
'''
    I_Dis = np.stack([ I_Targets[y[1]] for y in y_train])
    
    I_X_Train , Train_Targets = processImages(I_X_Train , Train_Targets)
    
    gc.collect()



    input("a")


    dis = get_disassociated(Y_idxs , len(Targets))
    
    Y_D_Images = [ (Targets[idx], Targets[d]) for idx , d in zip(Y_idxs , dis) ]

    Mod = PLDTGAN(X[0].shape , 64 , 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X , Y_D_Images )

    X_train = np.stack(X_train)

    y_train =  np.stack([y[0] for y in y_train])
    D_images = np.stack([y[1] for y in y_train])
    print(D_images.shape)
    X_train = np.array_split(X_train, 128)
    y_train = np.array_split(y_train , 128)
    D_images = np.array_split(D_images , 128)
    
    Mod.train(X_train , y_train , D_images)
    
    input("a")
    #Mod.train(X , Y_D_Images)
    
    #plot_model(Mod.GAN , show_shapes=True , to_file="model.png" )


    
    #Mod.train(X , Y_D_Images)
    print("Here")

    a = Mod.GAN(np.stack(X[:64]))
    
    print(a)
    
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