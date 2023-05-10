# Antillia.com Toshiyuki Arai
# 2023/05/10

# loss_function.py
#
# These functions have been taken from the following web site.
#
# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
#   MIT license

import tensorflow as tf
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  # 2023/05/10 Added casting to 'float32'
  y_true_f = K.cast(y_true_f, 'float32')
  y_pred_f = K.cast(y_pred_f, 'float32')
  intersection = K.sum(y_true_f * y_pred_f, axis=[1,2,3])
  union = K.sum(y_true_f, axis=[1,2,3]) + K.sum(y_pred_f, axis=[1,2,3])
  dice  = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def jacard_similarity(y_true, y_pred):
    """
    Intersection-Over-Union (IoU), also known as the Jaccard Index
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # 2023/05/10 Added casting to 'float32'
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
    return intersection / union

def jacard_loss(y_true, y_pred):
    """
    Intersection-Over-Union (IoU), also known as the Jaccard loss
    """
    return 1 - jacard_similarity(y_true, y_pred)