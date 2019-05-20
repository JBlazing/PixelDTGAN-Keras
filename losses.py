from tensorflow.math import log

def Assoc_Discrm_Loss(pred , t):


    
    return (-t * log(pred)) + ((t-1.0) * log(1.0 - pred))


def GANLoss(D , A):

    return -0.5 * (D + A) 

