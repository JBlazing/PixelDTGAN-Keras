import cv2
import models
import fileLoader
def main():

    #GenModel , DiscModel , AModel = models.createModels((64,64,2))

    files = fileLoader.getFiles('lookbook/data/')
    print(files)



if __name__ == "__main__":
    main()