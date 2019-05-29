from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from tensorflow import keras , convert_to_tensor , reshape, fill , shape
from tensorflow.data import Dataset
from tensorflow import GradientTape , reduce_mean
from losses import Assoc_Discrm_Loss , GANLoss
import numpy as np
from fileLoader import loadFiles , processImages
from tensorflow.math import log
from tensorflow.train import GradientDescentOptimizer
 
def createLayers(input , outputSize , kernel_size=(4,4), strides=2 ,leaky=True , batch=True , padding="same"):
    l = Conv2D(outputSize ,  kernel_size = kernel_size ,  strides=strides, padding=padding )(input)
    if leaky:
        l = LeakyReLU(.2)(l)
    if batch:
        l = BatchNormalization()(l)
    return l

#Generate label for target Y for the loss functions
#If 1 is generated then the target image was associated image was choosen

def labelGen(i):
    if i == 1:
        return 1.0
    return 0.0

class PLDTGAN:

    def __init__(self , input_shape , filters=64 , Depochs=10 , Gepochs=64 , batch_size=128):
        self._Depochs = Depochs
        self._Gepochs = Gepochs
        self._num_filters = filters
        self._input_shape = input_shape
        self._cutOff = 10
        self.batch_size = batch_size
        self.GAN = self.createGAN(input_shape , self._num_filters)
        self.Discrm = self.createDisc(input_shape , self._num_filters)
        self.Assoc = self.createAssociated(input_shape , filters)
    

    def train(self, X, Y , Targets):

        flag = True
        opt = GradientDescentOptimizer(learning_rate=1e-3)
        
        self.TrainDis_Ass(X,Y , Targets , opt)
       
        
        
     
        #self.TrainGAN(X,Y,Targets , opt)

        
        
        print("Done")
        return
   

    def TrainGAN(self, X,Y,Targets , opt):
        
        for epoch in range(self._Gepochs):
            print("Step %i of %i" % (epoch , self._Gepochs))
            for step , (x_batch , y_batch) in enumerate(zip(X,Y)):

                x_batch = loadFiles(x_batch)
                processImages(x_batch)
                tFiles = Targets[y_batch[: , 0]]
                dFiles = Targets[y_batch[: , 1]]
                
                y = loadFiles(tFiles)
                d = loadFiles(dFiles) 
                processImages(y)
                processImages(d)

                tt = [] 
                yy = []

                with GradientTape() as tape:
                    logits = self.GAN(x_batch)
                  
                  
                    tt = [ 0.0 for x in logits]
                    DY = self.Discrm(logits)
                
                    AY = self.Assoc([x_batch, logits])
                    
                    
                    #tt = reshape(DY , shape(DY))
                    DY  = reshape(DY , [-1])
                    AY = reshape(DY , [-1])
                    
                    Dloss_value = keras.losses.binary_crossentropy(tt , DY )
                    Aloss_value = keras.losses.binary_crossentropy(tt , AY )


                    
                    loss = GANLoss(Aloss_value , Dloss_value)
                    loss = reduce_mean(loss)
                    
                
                        
                    
                    #loss = fill(log.shape , loss)
                
                
                grads = tape.gradient( loss, self.GAN.trainable_variables)   
                
                
                print(grads , flush=True)
                #opt.apply_gradients(zip(grads , self.GAN.trainable_variables))


    def TrainDis_Ass(self , X , Y , Targets , opt):
        print("Starting The Associated/Discrm Training", flush=True)
        for epoch in range(self._Depochs):
            
            print("Step %i of %i" % (epoch , self._Depochs))
            for step, (x_batch, y_batch)  in enumerate(zip(X,Y)):
                

                x_batch = loadFiles(x_batch)
                processImages(x_batch)
                tFiles = Targets[y_batch[: , 0]]
                dFiles = Targets[y_batch[: , 1]]
                
                y = loadFiles(tFiles)
                d = loadFiles(dFiles) 
                processImages(y)
                processImages(d)

                tt = [] 
                yy = []

                logits = self.GAN(x_batch)
                
                #split out the target and disasociated images
                # and push either the apporiate image
                for i, (a , dis) in enumerate(zip(y,d)):
                    dec = np.random.randint(3)
                    if dec == 0:
                        yy.append(logits[i])
                    elif dec == 1:
                        yy.append(a)
                    else:
                        yy.append(dis)
                    tt.append(labelGen(dec))
                

                tt = convert_to_tensor(tt)
            
                yy = np.stack(yy)

                with GradientTape() as DTape: 
                    DY = self.Discrm(yy)
                    #reshape(DY , [-1])
                    tt = reshape(tt , shape(DY))
                    #loss_value = Assoc_Discrm_Loss(DY , tt )
                    Dloss_value = keras.losses.binary_crossentropy(DY , tt )
                    #print(Dloss_value)
                    #Dloss_value =  reduce_mean(Dloss_value)
                grads = DTape.gradient(Dloss_value , self.Discrm.trainable_variables)
                
                opt.apply_gradients(zip(grads,self.Discrm.trainable_variables))

                with GradientTape() as ATape:
                    #need to cat the images
                    
                    AY = self.Assoc([x_batch, yy])
                    #reshape(AY , [-1])
                    #loss_value = Assoc_Discrm_Loss(AY , tt)
                    Aloss_value = keras.losses.binary_crossentropy(AY , tt)
                   # Aloss_value = reduce_mean(Aloss_value)
                Agrads = ATape.gradient(Aloss_value , self.Assoc.trainable_variables )
                opt.apply_gradients(zip(Agrads,self.Assoc.trainable_variables)) 
        


    def test(self, x,y):
        pass

    '''
        Helper Functions to create network
    '''   
    def oldTrain(self, X, Y , Targets):

        flag = True
        opt = keras.optimizers.SGD(learning_rate=1e-3)
    
        print("Starting The Training", flush=True)
        for epoch in range(self._epochs):
            
            print("Step %i of %i" % (epoch , self._epochs))
            for step, (x_batch, y_batch)  in enumerate(zip(X,Y)):
                
              

                x_batch = loadFiles(x_batch)
                processImages(x_batch)
                tFiles = Targets[y_batch[: , 0]]
                dFiles = Targets[y_batch[: , 1]]
                
                y = loadFiles(tFiles)
                d = loadFiles(dFiles) 
                processImages(y)
                processImages(d)

                tt = [] 
                yy = []
                with GradientTape() as GTape:
                    
                    
                    logits = self.GAN(x_batch)
                    

                    #split out the target and disasociated images
                    # and push either the apporiate image
                    for i, (a , dis) in enumerate(zip(y,d)):
                        dec = np.random.randint(3)
                        if dec == 0:
                            yy.append(logits[i])
                        elif dec == 1:
                            yy.append(a)
                        else:
                            yy.append(dis)
                        tt.append(labelGen(dec))
                    

                    tt = convert_to_tensor(tt)
                
                    yy = np.stack(yy)

                    with GradientTape() as DTape: 
                        DY = self.Discrm(yy)
                        #reshape(DY , [-1])
                        
                        #loss_value = Assoc_Discrm_Loss(DY , tt )
                        Dloss_value = keras.losses.binary_crossentropy(DY , tt )
                        #print(Dloss_value)
                        Dloss_value =  reduce_mean(Dloss_value)
                    if epoch < self._cutOff:
                    
                        grads = DTape.gradient(Dloss_value , self.Discrm.trainable_variables)
                    
                        opt.apply_gradients(zip(grads,self.Discrm.trainable_variables))

                    with GradientTape() as ATape:
                        #need to cat the images
                        
                        AY = self.Assoc([x_batch, yy])
                        #reshape(AY , [-1])
                        #loss_value = Assoc_Discrm_Loss(AY , tt)
                        Aloss_value = keras.losses.binary_crossentropy(AY , tt)
                        Aloss_value = reduce_mean(Aloss_value)
                    
                    if epoch < self._cutOff:
                        Agrads = ATape.gradient(Aloss_value , self.Assoc.trainable_variables )
                        opt.apply_gradients(zip(Agrads,self.Assoc.trainable_variables)) 

                    loss_value = GANLoss(Dloss_value, Aloss_value)
                    #print(loss_value)

                if epoch >= self._cutOff:   
                    Ggrads = GTape.gradient(loss_value , self.GAN.trainable_variables)
                #print(self.GAN.trainable_variables , flush=True)
                    try:
                        opt.apply_gradients(zip(Ggrads,self.GAN.trainable_variables))
                    except:
                        print("it broke" , flush=True)
                        continue
        print("Done")
        return
    
    
    def newGAN(self , input_shape , filters):
        
        def createOutGenLayer(inLayer , outputSize , activation='relu' , norm=False , kernel_size=(4,4) , strides=(2,2)):
            l = Conv2DTranspose(outputSize , activation=activation , kernel_size=kernel_size , strides=strides , padding="same")(inLayer)
            if norm:
                l = BatchNormalization()(l)
            return l
        
        in_layer = Input(shape=input_shape , name = "Input")
        L1 = createLayers(in_layer , 3)
        #out = createOutGenLayer(L1 , 3)
        
        mod = Model(in_layer , L1)
        return mod
    
    def createGAN(self , input_shape , filters):

        def createOutGenLayer(inLayer , outputSize , activation='relu' , norm=True , kernel_size=(4,4) , strides=(2,2)):
            l = Conv2DTranspose(outputSize , activation=activation , kernel_size=kernel_size , strides=strides , padding="same")(inLayer)
            if norm:
                l = BatchNormalization()(l)
            return l
        
        in_layer = Input(shape=input_shape , name = "Input")

        L1 = createLayers(in_layer , filters)
        L2 = createLayers(L1 , filters * 2)
        L3 = createLayers(L2 , filters * 4)
        L4 = createLayers(L3 , filters * 8)
        L5 = createLayers(L4 , 100 , kernel_size=4 , strides=4)
        G5 = createOutGenLayer(L5 , filters * 4 , strides=4)
        G6 = createOutGenLayer(G5 , filters * 2)
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
        L4 = createLayers(L3 , filters * 8)
        L5 = createLayers(L4 , 1 , kernel_size=4, strides=4 ,leaky=False , batch=False)
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
        L4 = createLayers(L3 , filters * 8)
        
        L5 = createLayers(L4 , 1 ,  kernel_size=4, strides=4 , leaky=False , batch=False)
        L6 = Activation('sigmoid')(L5)
        
        AssocModel = Model(inputs=[image1,image2] , outputs=L6 )
        #AssocModel.compile(loss='mean_squared_error', optimizer='sgd')
        
        return AssocModel