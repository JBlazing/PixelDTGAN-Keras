import cv2
import models
import fileLoader
def main():

    #

    files = fileLoader.getFiles('lookbook/data/')
    GenModel , DiscModel , AModel = models.createModels((64,64,2))



if __name__ == "__main__":
    main()