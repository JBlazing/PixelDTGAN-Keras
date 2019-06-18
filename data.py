from sklearn.model_selection import train_test_split
from random import shuffle
import numpy as np
import glob
import cv2
from fileLoader import resizeImages


new_size = (256 , 64)

resizeImages('data_road/training/gt_pixel_cp/' , 'data_road/resized/training/gt_pixel_cp/' , new_size)


'''
img_path = "data_road/training/image_2/{}_{}"

gts = glob.glob("data_road/training/gt_image_2/*.png")

imgs = []

for gt in gts:

    file = gt.split("/")[-1]
    parts = file.split('_')
    img = '{}_{}'.format(parts[0],parts[2])
    imgs.append(img)
    
    
cpy = [ gt for gt in gts ]

f = open("training.csv" , "w")
for img , gt , dis in zip(imgs , gts , cpy):

    gt_splits = gt.split('/')
    dis_splits = dis.split('/')
    
    w = '{},{},{}\n'.format(img , gt_splits[-1] , dis_splits[-1] )
    
    f.write(w)
    
    
f.close()

filePaths = ['data_road/training/image_2/' , 'data_road/training/gt_image_2/' , 'data_road/training/gt_image_2/']
filePaths = ['data_road/resized/training/image_2/' , 'data_road/resized/training/gt_image_2/' , 'data_road/resized/training/gt_image_2/']
outPaths = ['data_road/resized/training/image_2/' , 'data_road/resized/training/gt_image_2/']
f = open('data_road/training.csv' , 'r')
    
b = f.readlines()

f.close()

b = [ l.split(',') for l in b ]

for l in b:
    l[-1] = l[-1].rstrip()

for l in b:
    for i in range(len(filePaths)):
        l[i] = filePaths[i] + l[i]
        
b = np.array(b)




sizes = set()

for row  in b:
    for im , path in zip(row , outPaths):
        print(im)
        img = cv2.imread(im)

        
        img = cv2.resize(img , new_size)
        outPath = '{}{}'.format(path,im.split('/')[-1] )
        cv2.imwrite(outPath , img)
        
'''


