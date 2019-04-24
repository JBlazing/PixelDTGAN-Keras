import cv2
import os
import re



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
        
        
    return [ itm for key, itm in prods.items()]
    

