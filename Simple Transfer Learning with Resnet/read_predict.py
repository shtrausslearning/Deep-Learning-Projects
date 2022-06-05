import pandas as pd
import numpy as np
from pathlib import Path,PurePath

# Import ML modules
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Instantiate ResNet Classifier w/ imagenet nn model weights
model = ResNet50(weights='imagenet')

# Visualise Model Summary
# model.summary()

# Function to load image & image is converted to a NumPy array
# The ResNet50 model expects a (224,224) sized image

def load_img(path):
    img = image.load_img(path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    return x

# Image from local directory
file = Path('/kaggle/input/flower-color-images/flower_images/flower_images/0002.png')
img = load_img(file)

# Model Prediction
pred = model.predict(img) # Outputs show class probabilities

# Decode the results into tuple list (class, description, probability)
print(decode_predictions(pred,top=3)[0])
