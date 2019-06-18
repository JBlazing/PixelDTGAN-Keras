from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow import keras , convert_to_tensor , reshape, fill , shape
from tensorflow.data import Dataset
from tensorflow import GradientTape , reduce_mean
from losses import Assoc_Discrm_Loss , GANLoss
import numpy as np
from fileLoader import loadFiles , processImages , deNormalize
from tensorflow.math import log

import cv2

def createLayers(input , outputSize , kernel_size=(4,8), strides=(2,2) ,leaky=True , batch=True , padding="same"):
    l = Conv2D(outputSize ,  kernel_size = kernel_size ,  strides=strides, padding=padding , kernel_initializer=keras.initializers.RandomNormal(stddev=0.02) )(input)
    if leaky:
        l = LeakyReLU(.2)(l)
    if batch:
        l = BatchNormalization()(l)
    return l

class PLDTGAN:

    def __init__(self , input_shape , filters=64 , epochs=25 ,  checkpoint=0):
        self._epochs = epochs + checkpoint
        self._num_filters = filters
        self._input_shape = input_shape
        self._cutOff = 10
        
        
        if checkpoint == 0:
            self.GAN = self.createGAN(input_shape , self._num_filters)
            self.Discrm = self.createDisc(input_shape , self._num_filters)
            self.Assoc = self.createAssociated(input_shape , filters)
        else:
            self.loadModels(checkpoint)

    def train(self, batches):

        flag = True
        self.opt = keras.optimizers.SGD(learning_rate=.0002 , momentum=.5)
        print("Starting The Associated/Discrm Training", flush=True)
    
         
        for epoch in range(1 , self._epochs + 1):
            print("Epoch %i of %i" % (epoch , self._epochs))
            A_total = 0.0
            D_total = 0.0 
            G_total = 0.0
            for step , x_batch in enumerate(batches):
                
                if step % 25 == 0:
                    print("Epoch {} | Batch {} of {}".format(epoch , step , len(batches)))
                
                lreal = np.full(len(x_batch) , 1.0)
                lfake = np.full(len(x_batch) , 0.0)
                
                input = loadFiles(x_batch[: , 0])
                processImages(input)
                tFiles = x_batch[ : , 1]
                dFiles = x_batch[: , 2]
                assoc = loadFiles(tFiles)
                disassoc = loadFiles(dFiles) 
                processImages(assoc)
                processImages(disassoc)               
                
                fake = self.GAN(input)
                
                D_Loss = self.DiscrmTrain(assoc , disassoc , fake , lreal , lfake)
                
                A_Loss = self.AssocTrain(input , assoc , disassoc , fake, lreal , lfake )
                
                G_Loss = self.GANTrain(input , lreal)
                
                A_total += reduce_mean(A_Loss)
                A_total += reduce_mean(D_Loss)
                G_total += reduce_mean(G_Loss)
            
                
        
            self.saveModels(epoch)

       
        
    def GANTrain(self, input , labels ):
        
        with GradientTape() as tape:
            
            fake = self.GAN(input)
            
            d_out = self.Discrm(fake)
        
            
            a_out = self.Assoc([input , fake])
            
            a_loss = keras.losses.binary_crossentropy(labels , a_out)
            
            d_loss = keras.losses.binary_crossentropy(labels, d_out )
            
            t_loss = (a_loss + d_loss) / 2.0

            grads = tape.gradient(t_loss , self. GAN.trainable_variables)
        
        self.opt.apply_gradients(zip(grads , self.GAN.trainable_variables))
        
        return t_loss
    
    def AssocTrain(self ,input ,  assoc , disassoc , fake , lreal , lfake):
    
        with GradientTape() as tape:
            
            a_out = self.Assoc([input , assoc])
            
            d_out = self.Assoc([input,disassoc])
            
            f_out = self.Assoc([input,fake])
            
            a_loss = keras.losses.binary_crossentropy(lreal , a_out )
            
            d_loss = keras.losses.binary_crossentropy(lfake , d_out )
            
            f_loss = keras.losses.binary_crossentropy(lfake, f_out )
            
            t_loss = (a_loss + d_loss + f_loss )/ 3.0
            
            grads = tape.gradient(t_loss , self.Assoc.trainable_variables)
            
        self.opt.apply_gradients(zip(grads , self.Assoc.trainable_variables))
        
        return t_loss
        
    def DiscrmTrain(self , assoc , disassoc , fake, lreal , lfake):
        
        with GradientTape() as tape:
            
            a_out = self.Discrm(assoc)
            
            d_out = self.Discrm(disassoc)
            
            f_out = self.Discrm(fake)
            
            a_loss = keras.losses.binary_crossentropy(lreal , a_out)
            
            d_loss = keras.losses.binary_crossentropy(lreal , d_out)
            
            f_loss = keras.losses.binary_crossentropy(lfake , f_out)
            
            t_loss = (a_loss + d_loss + f_loss )/ 3
            
            grads = tape.gradient(t_loss , self.Discrm.trainable_variables)
            
        self.opt.apply_gradients(zip(grads , self.Discrm.trainable_variables))
        
        return t_loss
        
    def test(self, batches):
        
        for batch in batches:
            
            imgs = loadFiles(batch)
            
            processImages(imgs)
            
            output = self.GAN(imgs)
            
            output = output.numpy()
            
            deNormalize(output)
            deNormalize(imgs)
            
            for file, i , o in zip(batch , imgs  , output):
                file = file.split('/')[-1]
                print(file)
                out = np.concatenate((i , o) , axis=0)
                cv2.imwrite("outputs/" + file, out)
        
    '''
        Helper Functions to create networks
    '''    
    def createGAN(self , input_shape , filters):

        def createOutGenLayer(inLayer , outputSize , activation='relu' , norm=True , kernel_size=(4,8) , strides=(2,2)):
            l = Conv2DTranspose(outputSize , activation=activation , kernel_size=kernel_size , strides=strides , padding="same" , kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(inLayer)
            if norm:
                l = BatchNormalization()(l)
            return l
        
        in_layer = Input(shape=input_shape , name = "Input")

        L1 = createLayers(in_layer , filters)
        L2 = createLayers(L1 , filters * 2)
        L3 = createLayers(L2 , filters * 4)
        L4 = createLayers(L3 , filters * 8 , strides=(4,8))
        L5 = createLayers(L4 , 200 ,  strides=(2,4)  )
        G5 = createOutGenLayer(L5 , filters * 4 , strides=(4,8))
        G6 = createOutGenLayer(G5 , filters * 2 , strides=(2,4) )
        G7 = createOutGenLayer(G6 , filters)
        G8 = createOutGenLayer(G7 , filters)
        
        G9 = createOutGenLayer(G8 , 3 , strides=2  ,activation='tanh' , norm=False)

        GenModel = Model(inputs=[in_layer] , outputs=[G9])
        #GenModel.compile(loss='mean_squared_error', optimizer='sgd')

        return GenModel

    def createDisc(self , input_shape , filters ):

        in_layer = Input(shape=input_shape , name="Discrm_Input")
        L1 = createLayers(in_layer , filters)
        L2 = createLayers(L1 , filters * 2)
        L3 = createLayers(L2 , filters * 4)
        L4 = createLayers(L3 , filters * 8, strides=(4,8))
        L5 = createLayers(L4 , 1 , strides=(2,4) ,leaky=False , batch=False)
        L6 = Activation('sigmoid')(L5)
        
        DiscModel = Model(inputs=[in_layer] , outputs=[L6])
       #DiscModel.compile(loss='mean_squared_error', optimizer='sgd')
        

        return DiscModel

    def createAssociated(self , inputs , filters):

        image1 = Input(shape=inputs , name="Image_1")       
        image2 = Input(shape=inputs , name="Image_2")       
        
        InCat = Concatenate()([image1 , image2])
        L1 = createLayers(InCat , filters)
        L2 = createLayers(L1 , filters * 2)
        L3 = createLayers(L2 , filters * 4)
        L4 = createLayers(L3 , filters * 8, strides=(4,8))
        
        L5 = createLayers(L4 , 1 , strides=(2,4) , leaky=False , batch=False)
        L6 = Activation('sigmoid')(L5)
        
        AssocModel = Model(inputs=[image1,image2] , outputs=L6 )
        #AssocModel.compile(loss='mean_squared_error', optimizer='sgd')
        
        return AssocModel
        
        
    def saveModels(self, epoch):
        print("Saving at Epoch {}".format(epoch))
        self.GAN.save("/media/hdd/checkpoints/GAN_{}_checkpoint.h5".format(epoch))
        self.Discrm.save("/media/hdd/checkpoints/Discrm_{}_checkpoint.h5".format(epoch))
        self.Assoc.save("/media/hdd/checkpoints/Assoc_{}_checkpoint.h5".format(epoch))
        
        
    def loadModels(self , epoch):
        self.GAN = keras.models.load_model("/media/hdd/checkpoints/GAN_{}_checkpoint.h5".format(epoch))
        self.Discrm = keras.models.load_model("/media/hdd/checkpoints/Discrm_{}_checkpoint.h5".format(epoch))
        self.Assoc = keras.models.load_model("/media/hdd/checkpoints/Assoc_{}_checkpoint.h5".format(epoch))