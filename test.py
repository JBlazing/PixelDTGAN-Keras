from fileLoader import getDataPaths
from models import PLDTGAN
import glob
import numpy as np
import tensorflow as tf
def main():

    imgs = glob.glob('data_road/resized/testing/*.png')
    batches = np.array(imgs)
    batches = np.array_split(batches , 28)
    
    print(len(batches))
    with tf.device('/device:GPU:0'):
        mod = PLDTGAN(None , checkpoint=57)
        
        #print(mod.GAN.summary())
        
        mod.test(batches)
    
    
if __name__ == "__main__":
    main()