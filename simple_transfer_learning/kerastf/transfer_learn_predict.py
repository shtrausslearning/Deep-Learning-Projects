import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions 
from tensorflow.keras import utils
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam

import warnings; warnings.filterwarnings("ignore")
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# After we have read the images from the folders & stored in np for at X,y

# Number of classes
n_classes = 2

# Load the model (only bottom part)
model = ResNet50(include_top = False, 
                 weights = 'imagenet', 
                 input_shape = (224,224,3))

# Building our classifier 
av1 = GlobalAveragePooling2D()(model.output)       # reduce size 
fc1 = Dense(256, activation = 'relu')(av1)         
d1 = Dropout(0.5)(fc1)                             # for generalisation 
fc2 = Dense(n_classes, activation = 'softmax')(d1) # probability of each class

# Our classification model
model_new = Model(inputs = model.input, 
                  outputs = fc2)

''' Compiling the new model '''
# Fine tuning our model 

adam = Adam(lr = 0.00003)                            # optimiser
model_new.compile(loss = 'categorical_crossentropy', 
                  optimizer = adam, 
                  metrics = ['accuracy'])

''' Training the model '''
# train model on some data (X_train, y_train)
# set non trainable layers

def set_nontrainable(n):
    
    # show layer index
    for ii in range(len(model_new.layers)):
        print(ii, model_new.layers[ii])
        
    for ii in range(n):
        model_new.layers[ii].trainable = False
        print(model_new.summary())

# Train the classifier
hist = model_new.fit(X, y, 
                     shuffle = True, 
                     batch_size = 2,
                     epochs = 5)
