# Antillia.com Toshiyuki Arai
# 2023/05/12

# losses.py
#
# These functions have been taken from the following web site.
#
# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
# https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
#   MIT license

"""
Orignal usage example taken from the website above is the follwing:
model.compile(optimizer=Adam(lr=1e-3),
              loss=semantic_loss.unet3p_hybrid_loss,
              metrics=[semantic_loss.dice_coef, semantic_loss.sensitivity, semantic_loss.specificity])

              
We use the following simplified version.
-->
model.compile(optimizer = Adam(lr=1e-3),
              loss      = basnet_hybrid_loss,
              metrics  = [dice_coef],
              or
              #metrics   = [dice_coef, sensitivity, specificity]
              )
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, BinaryCrossentropy


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = K.cast(y_true_f, 'float32')
    y_pred_f = K.cast(y_pred_f, 'float32')

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
   
 
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss / 2.0

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

def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) loss
    """
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')

    return 1 - tf.image.ssim(y_true_f, y_pred_f, max_val=1)

def basnet_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in BASNET (https://arxiv.org/pdf/2101.04704.pdf)
    The hybrid loss is a combination of the binary cross entropy, structural similarity
    and intersection-over-union losses, which guide the network to learn
    three-level (i.e., pixel-, patch- and map- level) hierarchy representations.
    """
    bce_loss = BinaryCrossentropy(from_logits=False)
    bce_loss = bce_loss(y_true, y_pred)

    ms_ssim_loss = ssim_loss(y_true, y_pred)
    jac_loss     = jacard_loss(y_true, y_pred)

    return bce_loss + ms_ssim_loss + jac_loss
