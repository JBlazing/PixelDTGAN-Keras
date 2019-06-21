from models import PLDTGAN
from fileLoader import getDataPaths
import cv2
import tensorflow as tf

def main():
    
    filePaths = ['data_road/resized/training/image_2/' , 'data_road/resized/training/gt_image_cp/' , 'data_road/resized/training/gt_image_cp/']
    batches = getDataPaths('data_road/new_training.csv' , filePaths , 16)
    
    size = cv2.imread(batches[0][0][0]).shape
    
    with tf.device('/device:GPU:1'):
        Mod = PLDTGAN(size , epochs=512 , checkpoint=256)
        
        #print(Mod.GAN.summary())
        
        Mod.train(batches)
        
        Mod.saveModels(Mod._epochs)

 
if __name__ == "__main__":
    main()