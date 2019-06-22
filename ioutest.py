import tensorflow as tf
from fileLoader import getDataPaths , loadFiles , processImages , deNormalize
import os
import re
import numpy as np
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)

Model_Path = '/media/hdd/checkpoints/'

regex = re.compile(r'GAN_*')

filePaths = ['data_road/resized/training/image_2/' , 'data_road/resized/training/gt_image_cp/' , 'data_road/resized/training/gt_image_cp/']

classes = np.array([[1.0,1.0,1.0] ,[1.0,-1.0,1.0]] , dtype=np.float32)




def pixel_To_Label(img , output ,classes):
    
    for i , row in enumerate(img):
        for j , col in enumerate(row):
           
            out = np.argmin( [ np.linalg.norm(col - c)  for c in classes ] )
            #print(out)
            output[i,j] = out
       

def main():
    
#   Get the latest model in the checkpoint folder
    
    Mods = os.listdir(Model_Path)
    sel = filter(regex.search , Mods)
    mod_file = max(sel , key=lambda a: int(a.split('_')[1]))
    mod_file = '{}{}'.format(Model_Path, mod_file)

    batches = getDataPaths('data_road/testing.csv' , filePaths , 32)

    predLabels = []
    actualLabels = []
    metric = tf.keras.metrics.MeanIoU(num_classes=2)
    with tf.device('/device:GPU:0'):
        Model = tf.keras.models.load_model(mod_file)
        
        
       
        for batch in batches:
            
            imgs  = loadFiles(batch[:,0])
            assocs = loadFiles(batch[:,1])
            
            processImages(imgs)
            processImages(assocs)
            
            preds = Model.predict_on_batch(imgs)
            
            #deNormalize(preds)
            
            t = None
            test = None
            
            predLabel = np.full(tuple(list(assocs.shape)[:-1]) , -1)
            actualLabel = np.full(tuple(list(assocs.shape)[:-1]) , -1)
            
            for i , (assoc , pred) in enumerate(zip(assocs , preds)):
                
                pixel_To_Label(pred , predLabel[i] , classes)
                pixel_To_Label(assoc , actualLabel[i] , classes)
                
                predLabels.append(predLabel[i])
                actualLabels.append(actualLabel[i])
                
                

            '''
            for assoc , pred in zip(assocs , preds):
            
                test = np.zeros(tuple(list(assoc.shape)[:2]))
                print(test.shape)
                for i , ( a_row , p_row )  in enumerate(zip(assoc , pred)):
                    
                    for j , (a_col , p_col) in  enumerate(zip(a_row , p_row)):
                        
                        np.array([ for x in ])
                        test[i,j] = np.linalg.norm(a_col - p_col)
                
                dis = np.amax(test, axis=1)
                t = np.argmax(test , axis=1)
                
                deNormalize(preds)
                deNormalize(assocs)
                for i , j in enumerate(t):
                
                    print(assoc[i,j], pred[i,j] , dis[i] , sep='\t')
            
                    
                break
            
            break
            '''
            
                
    for a , p in zip(actualLabels , predLabels):
        metric.update_state(a , p)
    print('Print Result: ' ,  metric.result().numpy())

if __name__ == '__main__':
    main()