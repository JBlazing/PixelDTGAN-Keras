from fileLoader import getDataPaths
from models import PLDTGAN

def main():

    batches = getDataPaths('lookbook/train.txt')

    mod = PLDTGAN(None , checkpoint=30)
    
    mod.test(batches)
    
    
if __name__ == "__main__":
    main()