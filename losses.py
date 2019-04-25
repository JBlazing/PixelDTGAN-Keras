from tensorflow.keras.backend import log

def Assoc_Discrm_Loss(pred , t):

    return (-t * log(pred)) + ((t-1) * log(1 - pred))


def GANLoss(D , A):

    return (-0.5 * D) - (0.5*A)

