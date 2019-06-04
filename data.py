from fileLoader import getFiles , parse_Filenames , get_disassociated , loadFiles , processImages
from sklearn.model_selection import train_test_split
import numpy as np

files = getFiles('lookbook/resized/')


X,Targets,Y_idxs = parse_Filenames(files)

X = np.array(X)


dis = get_disassociated(Y_idxs , len(Targets))
Targets = np.array(Targets)


X_train, X_test, y_train, y_test = train_test_split(X , np.array([ (x,y) for x, y in zip(Y_idxs , dis )]) , train_size=.8 )


X_Val ,  X_t , y_val , y_t  = train_test_split(X_test , y_test , train_size=.5)


def write_to_file(X , Y , T,  f):

    for x , y in zip(X , Y):

        string = "{},{},{}\n".format(x , T[y[0]] , T[y[1]])
        f.write(string)

f = open("lookbook/train.txt" , 'w')

write_to_file(X_train , y_train , Targets, f )

f.close()

f = open("lookbook/test.txt" , 'w')

write_to_file(X_t , y_t ,Targets, f)

f.close()

f = open("lookbook/val.txt" , "w")

write_to_file(X_Val , y_val ,Targets, f)

f.close()