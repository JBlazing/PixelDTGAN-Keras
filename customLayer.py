import numpy as np
from keras import backend as K
from keras.layers import Layer

class ImageLayer(Layer):

    def __init__(self, **kwargs):
        super(ImageLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        super(ImageLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        return np.random.choice(x)
