import cv2
from models import PLDTGAN
from fileLoader import getDataPaths
import numpy as np
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.python import debug as tf_debug
import tensorflow as tf

import gc

def main():
    
    batches = getDataPaths('lookbook/train.txt')
       
    Mod = PLDTGAN((64,64,3) , 128 , 30)
    
    Mod.train(batches)
    
    
    Mod.saveModels()

 
if __name__ == "__main__":
    main()