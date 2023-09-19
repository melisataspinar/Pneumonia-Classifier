# CS 464 Spring 2020 Term Project

from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import keras


gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth( gpu, True ) 
tf.config.set_soft_device_placement( True )


######################################################################
##  The below implementation of f-1 score is taken from 
##  https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
######################################################################
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
######################################################################
######################################################################

curdir = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters ----------------------------------------------------------------------
hp = {
    "learning_rate": 0.1,
	"l2": 10,
    "l1": 0.01,
    "dropout_rate": 0.2, 
	
    "batch_size": 32,
	"epochs" : 6,
	
    "train_size": 4716,
    "val_size": 516,
    "test_size": 624,
	
    "train_directory": (str(curdir) + "\\chest-xray-pneumonia\\chest_xray\\train"),
    "val_directory": (str(curdir) + "\\chest-xray-pneumonia\\chest_xray\\val"),
    "test_directory": (str(curdir) + "\\chest-xray-pneumonia\\chest_xray\\test")
}

# Global Variables --------------------------------------------------------------------
image_width = 256
image_height = 256
freezed_layers = -1

# Template / Pseudo Model -------------------------------------------------------------
pseudo_model = InceptionV3( input_shape = ( image_width, image_height, 3 ),
                            include_top = False, weights = "imagenet")

# Freezing the layers -----------------------------------------------------------------
#for idx, layer in enumerate( pseudo_model.layers ):
#    # Freeze the first (freezed_layers + 1) layers
#    if idx > freezed_layers:
#        break
#    layer.trainable = False

# Layers ------------------------------------------------------------------------------
last_output = pseudo_model.output

x = Flatten()(last_output)
x = Dense( 128, activation = "relu", kernel_regularizer = l2(hp["l2"]), bias_regularizer = l2(hp["l2"]) )(x)
x = Dropout(hp["dropout_rate"])(x)
x = Dense( 128, activation = "relu", kernel_regularizer = l2(hp["l2"]), bias_regularizer = l2(hp["l2"]) )(x)
x = Dropout(hp["dropout_rate"])(x)
x = BatchNormalization()(x, training = True )
x = Dense( 2, activation = "sigmoid")(x)

#  Using softmax activation in the last layer: 
#  an x-ray might belong to neither normal or pnemonia classes

# Creating & Training the model -------------------------------------------------------
model = Model( pseudo_model.input, x )
model.compile( loss = "categorical_crossentropy", optimizer = RMSprop( lr = hp["learning_rate"] ), metrics = ["acc", f1_m, AUC()] )
model.summary()
print( len( model.layers ) )
model.save("base_model.h5")