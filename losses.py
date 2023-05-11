# Antillia.com Toshiyuki Arai
# 2023/05/11

# losses.py
#
# These functions have been taken from the following web site.
#
# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
#   MIT license

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses

def dice_loss(y_true, y_pred):
    smooth = 1.

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1.0 - score

def bce_dice_loss(y_true, y_pred):
    loss = 0.5 * losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
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