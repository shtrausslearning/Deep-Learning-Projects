![](https://i.imgur.com/XilezGZ.png)

### 1 | Purpose of this study

- The aim of this project is to create a model that will be able to determine from <code>x-rays</code>:
  - Which patiets have been labeled **corona positive** (option 1) and which are **standard x-rays** (option 2)
- The model will need to find subtle patterns in the images, so we'll need to utilise <code>CNN</code> models & build a <code>binary classifier</code> 
- We will need to find both normal and corona positive <code>x-ray</code> images & utilise the <code>keras</code> deep learning module to classify images 

### 2 | Study Keywords 

<code>x-rays</code> <code>images</code> <code>self sorted folders</code> <code>keras</code> <code>CNN model</code> <code>flow_from_directory</code> <code>image augmentation</code>

### 3 | Assembling Dataset
- The dataset being used in this project have been combined from two different sources **[source 1](https://github.com/ieee8023/covid-chestxray-dataset)** **[source 2](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**
  - <code>Source 1</code> contains images from <code>x-rays</code> of patients who have been **labeled** to be corona positive patients
  - <code>Source 2</code> contains images from normal <code>x-rays</code> 

The two sources are used to create a one unified dataset

Image <code>resolution</code>:
- (224,224,3) [px]

The following <code>augmentations</code> were used:
- shear_range=0.2
- zoom_range=0.2
- horizontal_flip=True
- rescale=1/255

### 4 | Files of Study:
<code>main.py</code> - Training File

### 5 | Process:
- [1] First, we need to create a root folder, in which we will be placing our dataset
- [2] We need to then create a folder, which will contain both <code>train</code> & <code>validation</code> folders
- [3] For both <code>train</code> & <code>validation</code>, we need to decide a distribution (how much data is used in training & validation)
- [4] Once decided, <code>x-rays</code> will be sorted into two folders <code>covid</code> & <code>normal</code>
- [5] Run <code>main.py</code>, which will display the <code>accuracy</code> on both <code>training</code> & </code>validation</code> datasets

### Define a Model

```python

''' Create Model Architecture '''
# sequential model, Conv2D model w/ generalisations
# Binary classification output (activation='sigmoid')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu",
                 input_shape=(224,224,3)))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))  # for binary classification

```

### Compile Model

```python

''' Compile Model '''
# During training, evaluation metrics for accuracy are shown & stored in fit() output
# adam optimiser, binary cross entropy loss function

model.compile(loss="binary_crossentropy", 
              optimizer="adam",
              metrics = ["accuracy"])
# model.summary()

```

### Create Data Generators

```python

''' Create Data Generators '''
# Generate batches of tensor image data with real-time data augmentation.
train_datagen = image.ImageDataGenerator(rescale=1./255,  # used to rescale the data values
                                         shear_range=0.2, # specifies the shear angle counter-clockwise in deg
                                         zoom_range=0.2,  # specifies the range of zoom for an image
                                         horizontal_flip=True) # boolean value which tells whether to flip the image horizontally or not.
test_datagen = image.ImageDataGenerator(rescale = 1./255)

# Train/Validation Generators Ojbects
train_generator = train_datagen.flow_from_directory('dataset/Train',
                                                    target_size=(224,224),
                                                    batch_size=32, 
                                                    class_mode="binary")
val_generator = test_datagen.flow_from_directory('dataset/Val',
                                                 target_size=(224,224),
                                                 batch_size=32, class_mode="binary")
                                                 
```

### Train Model

```python

''' Train the Model '''

hist = model.fit(train_generator, 
                 validation_data=val_generator, 
                 epochs = 6, 
                 verbose=0,
                 validation_steps=2)

```

## Result:
- After 10 <code>epoch</code> (dataset passes), the <code>CNN</code> model accuracy for training was **0.9480** and **0.9667**

## Discussion:
- This was a rather small project, coding is minial & the dataset has been sorted and assembed for us
- The <code>CNN</code> we used showed some good results, generalising more than overfitting (partly due to the <code>dropout</code> layers)
- Having trained on a <code>train</code>/<code>validation</code> split, we cannot be fully confident in the result & more thorough <code>Cross Validation</code> is required
- <code>Ensembling</code> of different models, such as <code>pretrained</code> models often show promising results, so that would be a good next step
