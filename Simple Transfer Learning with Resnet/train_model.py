import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras.applications.resnet50 import preprocess_input

# [notes]

# Having include_top=True means that a fully-connected layer will be added at the end of the model. 
# This is usually what you want if you want the model to actually perform classification. 
# With include_top=True you can specify the parameter classes (defaults to 1000 for ImageNet). 
# With include_top=False, the model can be used for feature extraction, for example to build an autoencoder or to stack 
# any other model on top of it. Note that input_shape and pooling parameters should only be specified when include_top is False.

# Setting to False -> we will built our own classifiers on top of the convolutional base of the ResNet50 model.
# Specify input_shape when include_top is False

# When you print the summary of the model, you will see the (7 x 7 x 2048) sized output from the model. 
# This shows that after flattening, we would have a large number of parameters. 
# To avoid this, we apply GlobalAveragePooling before passing the vector to the Fully Connected Layer. 
# Check the output by running the code shown above.

''' Create our own Classifier '''
# Number of classes
n_classes = 5

# Load the model (only bottom part)
model = ResNet50(include_top = False, 
                 weights = 'imagenet', 
                 input_shape = (224,224,3))
# print(model.summary())

# Building our classifier 
av1 = GlobalAveragePooling2D()(model.output)
fc1 = Dense(256, activation = 'relu')(av1)
d1 = Dropout(0.5)(fc1)                             # for generalisation 
fc2 = Dense(n_classes, activation = 'softmax')(d1) # probability of each class

# Our classification model
model_new = Model(inputs = model.input, 
                  outputs = fc2)
model_new.summary()

''' Make Some Predictions '''
# using the existing model

def load_img(path):
    img = image.load_img(path,
                         target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    return x
  
# decode_predictions can't be used because we added
# our own classifier 
image_path = 'image.png'
image = load_img(path)
pred = model_new.predict(image)
print(np.argmax(pred))

''' Compiling the new model '''
# Fine tuning our model 

adam = Adam(lr = 0.00003)                            # optimiser
model_new.compile(loss = 'categorical_crossentropy', 
                  optimizer = adam, 
                  metrics = ['accuracy'])

''' Training the model '''

def set_nontrainable(n):
  
  # show layer index
  for ii in range(len(model_new.layers)):
      print(ii, model_new.layers[ii])

  # set non trainable layers
  for ii in range(n):
      model_new.layers[ii].trainable = False
  print(model_new.summary())

# Train the classifier
hist = model_new.fit(X_train, Y_train, 
                     shuffle = True, 
                     batch_size = 16, 
                     epochs = 8, 
                     validation_split = 0.2)
