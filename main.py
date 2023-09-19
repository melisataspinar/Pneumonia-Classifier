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
	"l2": 1000,
    "l1": 0.01,
    "dropout_rate": 0.2, 
	
    "batch_size": 16,
	"epochs" : 10,
	
    "train_size": 4716,
    "val_size": 516,
    "test_size": 624,
	
    "train_directory": (str(curdir) + "\\input\\train"),
    "val_directory": (str(curdir) + "\\input\\val"),
    "test_directory": (str(curdir) + "\\input\\test")
}

# Global Variables --------------------------------------------------------------------
image_width = 256
image_height = 256
freezed_layers = -1

# Image Generators --------------------------------------------------------------------
datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
	rotation_range = 40,
    zoom_range = 0.15,
	width_shift_range = 0.15,
    height_shift_range = 0.15,
	brightness_range= [0.8, 1.2],
	fill_mode = "nearest",
    horizontal_flip = True,
    vertical_flip = False
)

datagen_test = ImageDataGenerator(
	rescale = 1./255,
	horizontal_flip = False,
    vertical_flip = False
)

# Batch Generators --------------------------------------------------------------------
train_generator = datagen.flow_from_directory(
    hp["train_directory"],
	batch_size = hp["batch_size"],
    target_size = ( image_width, image_height ),
	shuffle = True,
    class_mode = "categorical",
    color_mode = "rgb"
)

test_generator = datagen_test.flow_from_directory(
    hp["test_directory"],
    batch_size = hp["batch_size"],
    target_size = ( image_width, image_height ),
	shuffle = True,
    class_mode = "categorical",
    color_mode = "rgb"
)

validation_generator = datagen.flow_from_directory(
    hp["val_directory"],
    batch_size = hp["batch_size"],
    target_size = ( image_width, image_height ),
	shuffle = True,
    class_mode = "categorical",
    color_mode = "rgb"
)

# Callback ----------------------------------------------------------------------------
test_accs = []

class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        self.best = np.Inf


    def on_epoch_end(self, epoch, logs={}):
        test_accu = self.model.evaluate( test_generator, steps=hp["test_size"]//hp["batch_size"], verbose = 1)
        test_accs.append(test_accu[1])
        print('The testing accuracy is :',test_accu[1]*100, '%')
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            best_weights[0] = (self.model.get_weights())

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

# Creating & Training the model -------------------------------------------------------
model = Model( pseudo_model.input, x )
model.compile( loss = "categorical_crossentropy", optimizer = RMSprop( lr = hp["learning_rate"] ), metrics = ["acc", f1_m, AUC()] )
model.summary()
print( len( model.layers ) )

model.load_weights(str(curdir) + '\\base_model.h5')
#print(model.get_weights())

best_weights = [model.get_weights()]

history1 = model.fit(
  train_generator,
  steps_per_epoch = hp["train_size"] // hp["batch_size"],
  epochs = hp["epochs"],
  validation_data = validation_generator,
  validation_steps = hp["val_size"] // hp["batch_size"],
  verbose = 2,
  callbacks=[MyCallback()]
)

# Printing out the parameters used ---------------------------------------------------
print( "\nDropout Rate: " + str(hp["dropout_rate"]) + "\nLearning Rate: " + str(hp["learning_rate"]) + "\nL2: " + str(hp["l2"]) + "\n")
print( "Freezed Layers: " + str(freezed_layers+1) + "\nImage Size: " + str(image_width) + " x " + str(image_height))

# Testing the model ------------------------------------------------------------------
model.set_weights(best_weights[-1])
test_accu = model.evaluate( test_generator, steps=hp["test_size"]//hp["batch_size"], verbose = 1)

dictionary = dict( zip( model.metrics_names, test_accu) )
print( dictionary )



# Plotting the graphs ----------------------------------------------------------------
plt.plot( history1.history['acc'])
plt.plot( history1.history['val_acc'])
plt.title( 'Model Accuracy' )
plt.legend( ['Training Set', 'Validation Set'], loc = 'upper left' )
plt.xlabel( 'Epoch' )
plt.ylabel( 'Accuracy' )
plt.show() 

plt.plot( history1.history['loss'])
plt.plot( history1.history['val_loss'])
plt.title( 'Loss' )
plt.legend( ['Training Set', 'Validation Set'], loc = 'upper left' )
plt.xlabel( 'Epoch' )
plt.ylabel( 'Loss' )
plt.show()

plt.plot( history1.history['acc'])
plt.plot( test_accs )
plt.title('Model Test Accuracy vs Train Accuracy')
plt.legend(['Training set', 'Test set'], loc = 'upper left')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Saving the model -------------------------------------------------------------------
model.save("model.h5")
