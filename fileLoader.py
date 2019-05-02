import cv2
import os
import re
import numpy as np
import sklearn as sk
def getFiles(path):
    
    files = os.fsencode(path)

    
    prods = {}
    for nam in os.listdir(files):
        name = nam.decode()
        parts = name.split("_")
        id = re.findall('\d+' , parts[0])
        id = int(id[0])
        #image = cv2.imread(path + os.fsdecode(nam))
        f_path = path + os.fsdecode(nam)
        
        if int(id) in prods:
            prods[id].append(f_path)
        else:
            prods[id] = [f_path]
        
    for key , itm in prods.items():
        itm.sort()
    return [ itm for key, itm in prods.items()]
    


def resizeImages(path , save_path , new_size):

    files = os.fsencode(path)
    
    for file in os.listdir(files):
        image = cv2.imread(path + os.fsdecode(file))
        image = cv2.resize(image , new_size)
        cv2.imwrite(save_path + os.fsdecode(file) , image)

def processImages(X,Y):

    
    return  (   [ cv2.normalize(x , None ,-1 , 1 , norm_type=cv2.NORM_MINMAX , dtype=cv2.CV_32F) for x in X] , \
                [ cv2.normalize(x , None ,-1 , 1 , norm_type=cv2.NORM_MINMAX , dtype=cv2.CV_32F) for x in Y] )

def parse_Filenames(filesList):

    X = []
    Y = []
    Y_Idxs = []
    y_index = 0
    for item in filesList:
        models = item[ :-1]
        target = item[-1]
        t = [ y_index for x in models]
        X += models
        Y.append(target)
        Y_Idxs += t
        y_index += 1

    return X,Y,Y_Idxs

def get_disassociated(Y_Idxs , N):

    dis = []
    for Y in Y_Idxs:
        while True:
            r = np.random.randint(0,N)
            if r != Y:
                break
        
        dis.append(r)

    return dis


def loadFiles(X , Y):

    return [ cv2.imread(x) for x in X ] , [ cv2.imread(y) for y in Y ]
