import numpy as np
from keras import backend as K
from keras.layers import Layer

class ImageLayer(Layer):
    
    def __init__(self, output_dim,  **kwargs):
        self.output_dim = output_dim
        super(ImageLayer, self).__init__(**kwargs)
    
        def build(self , input_shape):
            assert isinstance(input_shape, list)
            super(ImageLayer , self).build(input_shape)

        def call(self , i):
            assert isinstance(x , list)
            return i
            
        
        def compute_output_shape(self , input_shape ):
            assert isinstance(input_shape, list)
            shape_a, shape_b = input_shape
            print(input_shape , flush=True)
            return shape_b