import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions 
from tensorflow.keras import utils
from tensorflow.keras.applications.resnet50 import ResNet50
import warnings; warnings.filterwarnings("ignore")
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# import file
import urllib.request
url = 'https://i.imgur.com/SdGxz2D.jpg'
fpath = 'SdGxz2D.jpg'
urllib.request.urlretrieve(url, fpath)
img = Image.open(fpath)
img = img.resize((224, 224), Image.ANTIALIAS)

# Convert to Numpy
x = utils.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
print(f"image shape: {x.shape}")

# Instantiate ResNet Classifier w/ imagenet nn model weights
model = ResNet50(weights='imagenet')

# Model Prediction
pred = model.predict(x) # Outputs show class probabilities
print(f"predicted class: {np.argmax(pred)}")

print("decode image")
predictionLabel = decode_predictions(pred, top = 1)
predictionLabel
