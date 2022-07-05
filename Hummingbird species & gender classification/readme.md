![](https://i.imgur.com/icqghBp.png)

Project available on Kaggle: **[Hummingbird Classification | Keras CNN Models](https://www.kaggle.com/code/shtrausslearning/hummingbird-classification-keras-cnn-models)** <br>
Dataset available on Kaggle: **[Hummingbirds at my feeder](https://www.kaggle.com/datasets/akimball002/hummingbirds-at-my-feeders)** <br>

Project Keywords: <br>
<code>keras</code> <code>CNN</code> <code>multiclass</code> <code>classification</code> <code>augmentation</code> <code>dataset from folder</code> <code>pretrained</code> <code>inference</code>


### 1 | Hummingbird monitoring

- The purpose of this project is to create an image <code>classifier</code> for hummingbird **species** and **genders** that visit feeders. 
- Such a classifier should be applicable to anywhere that hummingbirds migrate or breed, given additional datasets for those species.
- Hummingbird migration is otherwise reliant on individual bird watchers to see and report their observations. 
- If avid bird lovers setup a similar system, then conservation organizations would have better information on migratory and breeding patterns. 
- This knowledge can be used to determine if specific changes in the environment or ecology has positively or negatively impacted bird life.

### 2 | Reasons for turning to machine learning

- With the increased affordability of visual monitoring equipment, it has become practical for anyone to contribute to such a wonderful cause & help make each sighting more valid. 
- Often, due to <b>limitations of gear</b>, <b>poor photography/videography technique</b>, or **<span style='color:#B6DA32'>simply poor lighting conditions</span>**, it can be difficult for users, <b>let alone experts</b> to distinguish what specific bird was seen at the time of monitoring.
- In this study, we will focus our attention to a bird called the <b>hummingbird</b>. What is useful about this specie is that, despite being quite distinguishible in shape, <b>they have a variery of unique colouring</b>, which means that if images are of poor quality, it may be hard for humans or models to distinguish them.
- In the entire process of <b>expert identification</b> & dealing with various image related inconsistencies outlied previously, manual identification for monitoring can be quite labourous, so an automated system of identification can go a long way.
- In our quest to create an automated approach, we can be left with a collection or under or over exposed images that will create difficulties for the model to distinguish between different classes correctly. 

### 3 | Project goal 

- The ultimate goal is to have a <code>classification system</code> that can address such the above stated varieties in an image & correctly distinguish very similar bird species, it should be deployable at any feeder, which is important to the continued monitoring of hummingbird species and bird  migration patterns. 

### 4 | Model Exploration

- In this study, we'll be looking at creating a few <code>classification</code> models:
  - Create a base convolution neural network (CNN) model & test its performance (**Section 7**)
  - Find the best augmentation combination using the same CNN model (**Section 8**)
  - Use the combination of <code>augmentations</code> that resulted in the best performance & test different pretrained models, find the best performing model (**Section 9**)
  - If the pretrained models outperform the base CNN model, use them for inference on unseen data (**Section 10**)

### 4 | Configuration File

- We'll create a simple configuration class, that will contain all the important control settings
- number of class labels, shape of the images, number of training epochs & seed value

```python

''' Global Configuration Settings '''
class CFG:
    
    def __init__(self):
        self.labels = 4
        self.sshape = (100,100,3) # Increasable to 224
        self.n_epochs = 50
        self.seed = 221

cfg = CFG()
```

### 5 | The Dataset

- The dataset contains a main folder; <code>hummingbirds</code>, which contains image data split into <code>training</code>, <code>validation</code> & <code>test</code> sets, so the <code>train/test</code> split has already been done for us

```python
''' Folder Pathways'''
main_folder = '/kaggle/input/hummingbirds-at-my-feeders/'
train_folder = '/kaggle/input/hummingbirds-at-my-feeders/hummingbirds/train/'
val_folder = '/kaggle/input/hummingbirds-at-my-feeders/hummingbirds/valid/'
test_folder = '/kaggle/input/hummingbirds-at-my-feeders/hummingbirds/test/'
video_folder = '/kaggle/input/hummingbirds-at-my-feeders/video_test/'
```

```python
os.listdir(main_folder)
```

```
['video_test', 'All_images', 'hummingbirds']
```

```python
os.listdir(train_folder)
```

```
['Rufous_female', 'No_bird', 'Broadtailed_female', 'Broadtailed_male']
```

- Let's check how many images there are in each folder <code>Rufous female</code>, <code>No bird</code>, <code>Broadtailed female</code> & <code>Broadtailed male</code>
- We have a total of **4 classes**, so we'll be treating the problem as as <code>multiclass</code> classification problem

```python

class_types = len(os.listdir(train_folder))
print('Number of classes for Classification: ',class_types)
class_names = os.listdir(train_folder)
print(f'The class names are {class_names}\n')

print('Training dataset:')
for i in class_names:
    print(i + ':' + str(len(os.listdir(train_folder+i))))

print('\nValidation dataset:')
for i in class_names:
    print(i + ':' + str(len(os.listdir(val_folder+i))))
    
print('\nTest dataset:')
for i in class_names:
    print(i + ':' + str(len(os.listdir(test_folder+i))))
    
```

- The class distribution is even, the <code>training</code> data containing **100 samples** per class
- Both <code>validation</code> and <code>test</code> data containing **20 samples** per class

```
Number of classes for Classification:  4
The class names are ['Rufous_female', 'No_bird', 'Broadtailed_female', 'Broadtailed_male']

Training dataset:
Rufous_female:100
No_bird:100
Broadtailed_female:100
Broadtailed_male:100

Validation dataset:
Rufous_female:20
No_bird:20
Broadtailed_female:20
Broadtailed_male:20

Test dataset:
Rufous_female:20
No_bird:20
Broadtailed_female:20
Broadtailed_male:20
```

### 6 | Image Exploration

- Using the function <code>show_grid</code>, we can visualise the various types of class images
- We have **four** classes, let's make some remarks about each class after visualising the sample data

```python

''' Visualise Image Data '''
# Visualise a certain number of images in a folder using ImageGrid

def show_grid(image_list,nrows,ncols,label_list=None,
              show_labels=False,savename=None,
              figsize=(20,10),showaxis='off'):
    
    if type(image_list) is not list:
        if(image_list.shape[-1]==1):
            image_list = [image_list[i,:,:,0] for i in range(image_list.shape[0])]
        elif(image_list.shape[-1]==3):
            image_list = [image_list[i,:,:,:] for i in range(image_list.shape[0])]
    fig = plt.figure(None, figsize,frameon=False)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     share_all=True)
    
    for i in range(nrows*ncols):
        ax = grid[i]
        img = Image.open(image_list[i])
        ax.imshow(img,cmap='Greys_r')  # The AxesGrid object work as a list of axes.
        ax.axis(showaxis)
        if show_labels:
            ax.set_title(class_mapping[y_int[i]])
    if savename != None:
        plt.savefig(savename,bbox_inches='tight')
        

```

#### **CLASS 1 : RUFOUS FEMALE**
- The addition of the <b>female as opposed to the male</b> is an interesting choice for the dataset, making it indeed very challenging for the model, due to the high similarity of different species' image values
- The female, unlike the male is very similar to the <b>broadtail female</b>, <b>especially when in the shade</b>, we actually have quite a few such cases, as seen in the images below
- However we can notice that in all images, colours of the back/rump are <b>quite dull, more saturated, but with hints of green</b>, just enough to be able to make out the green colour
- We can also note that the images have quite a <b>bit of "noise"</b>, that could potentally affect the accuracy of the model; <b>the feeder & the flag introduce red values into the images</b>, which can affect the model accuracy

![](https://www.kaggleusercontent.com/kf/99926830/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nnCBGmtM_lkYAJo4IIyhLg.McJxw2Cq5L0j1Awa_cuPV3Psv6rfmJugWJ7DSYJ14sWyAMnjmJ8KHNXqhn3AXcKWvScQLPRnw5P4o1hpdhaPii5_EFU3AYrA_MLHkwXPMUkShmCwxX6yJ8MhDmiMTOYw281AbJMj4-ErMSlrC4aX6UMuyHiGFG4Kdp1itfrVzL0L23GH2gs8g2xSKTUkw8jeJepReHRxvT8_z1TIPOznnytVSP9odtaaD8P_3G_-3NywZIPgVoBQdgm9cHIhgPDFois-IiWXKFxEmhWL5gsdu4UViFACNoBkhxh0IzSnvLqnBWTVcY6sTX8Ta1fEfa_SnlFMNk2ijBEJEzlhQQwOKew8kwlCH_tLNdn9m8U9q_9plL_BaI_scpdEVI0FQUPAhf7m6aYVXy2LZaC_GBaIXWsJTN0ZuGk2PdT57K3Q415ljMRx5ILBkmU9utbjGx3Ft8RVw_FKkxyhSmaPA8q4jtYxv-BUnWMbNwTDl_0PcQovCyyCCczqm2ByB5X4E8_BevMbsdfdOxtQf9yFrxnST0aS3rYuQ8rxXFW5XYsJb_X5BsLegggA9LI6kxuYtd_BO-pJarpsO0yhWmiPZ4dkU-7ziCwTM5DxsGCFK61mpLGmM3gb25cuCaz1UpnboRpZhXbKHiTLfm-6XL_sYDum2fuohiYeg4gLv7nkY198iiCejKhPVKcZ5wgN0CMZ3JmsgbwELyY1dGfOxNeeX4Z0tg.hyHM-Hzr4VkeEiFh8IEEcA/__results___files/__results___10_0.png)

#### **CLASS 2 : BROADTAIL FEMALE**
- To the naked eye, there is <b>a lot of similarities</b> between the <b>Rufous</b> & <b>Broadtail Females</b>
- Without adequate lighting and refence to multiple frames (such as from a video), one could easily mislabel the species
- The boadtail female colour definitely stand out more, the two species so far have very similar bagrounds & the ocassional feeder

![](https://www.kaggleusercontent.com/kf/99926830/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nnCBGmtM_lkYAJo4IIyhLg.McJxw2Cq5L0j1Awa_cuPV3Psv6rfmJugWJ7DSYJ14sWyAMnjmJ8KHNXqhn3AXcKWvScQLPRnw5P4o1hpdhaPii5_EFU3AYrA_MLHkwXPMUkShmCwxX6yJ8MhDmiMTOYw281AbJMj4-ErMSlrC4aX6UMuyHiGFG4Kdp1itfrVzL0L23GH2gs8g2xSKTUkw8jeJepReHRxvT8_z1TIPOznnytVSP9odtaaD8P_3G_-3NywZIPgVoBQdgm9cHIhgPDFois-IiWXKFxEmhWL5gsdu4UViFACNoBkhxh0IzSnvLqnBWTVcY6sTX8Ta1fEfa_SnlFMNk2ijBEJEzlhQQwOKew8kwlCH_tLNdn9m8U9q_9plL_BaI_scpdEVI0FQUPAhf7m6aYVXy2LZaC_GBaIXWsJTN0ZuGk2PdT57K3Q415ljMRx5ILBkmU9utbjGx3Ft8RVw_FKkxyhSmaPA8q4jtYxv-BUnWMbNwTDl_0PcQovCyyCCczqm2ByB5X4E8_BevMbsdfdOxtQf9yFrxnST0aS3rYuQ8rxXFW5XYsJb_X5BsLegggA9LI6kxuYtd_BO-pJarpsO0yhWmiPZ4dkU-7ziCwTM5DxsGCFK61mpLGmM3gb25cuCaz1UpnboRpZhXbKHiTLfm-6XL_sYDum2fuohiYeg4gLv7nkY198iiCejKhPVKcZ5wgN0CMZ3JmsgbwELyY1dGfOxNeeX4Z0tg.hyHM-Hzr4VkeEiFh8IEEcA/__results___files/__results___12_0.png)

#### **CLASS 3 : BROADTAIL MALE**
- Like the adult female, the males also have green and buffy flanks
- What separates the male broadtail from the female, and even from the Rufous female is the **distinctive rose/magenta throats**.
- It's quite likely the model would be easily able to classify any image containing the male from the rest
- We can clearly observe that the feeder, has both darker spots and lighter spots, lighter spots have values very similar values to the throat
- We can also note in some images, they don't have this distinctive red colour throat (at least to the naked eye), one possible reason being that the bird is <b>in the shade at the time of capture</b>. It's also possible that the images are not correctly labeled, which cannot also be ruled out. However we can clearly note that the model will need to adapt to images <b>taken under direct sunlight</b> and <b>different shade variations</b>, which create some problematic scenarios
- And <b>last but not least</b>, we can't rule out <b>immature males</b> from the pack as well. They are extremly similar to female as well

![](https://www.kaggleusercontent.com/kf/99926830/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nnCBGmtM_lkYAJo4IIyhLg.McJxw2Cq5L0j1Awa_cuPV3Psv6rfmJugWJ7DSYJ14sWyAMnjmJ8KHNXqhn3AXcKWvScQLPRnw5P4o1hpdhaPii5_EFU3AYrA_MLHkwXPMUkShmCwxX6yJ8MhDmiMTOYw281AbJMj4-ErMSlrC4aX6UMuyHiGFG4Kdp1itfrVzL0L23GH2gs8g2xSKTUkw8jeJepReHRxvT8_z1TIPOznnytVSP9odtaaD8P_3G_-3NywZIPgVoBQdgm9cHIhgPDFois-IiWXKFxEmhWL5gsdu4UViFACNoBkhxh0IzSnvLqnBWTVcY6sTX8Ta1fEfa_SnlFMNk2ijBEJEzlhQQwOKew8kwlCH_tLNdn9m8U9q_9plL_BaI_scpdEVI0FQUPAhf7m6aYVXy2LZaC_GBaIXWsJTN0ZuGk2PdT57K3Q415ljMRx5ILBkmU9utbjGx3Ft8RVw_FKkxyhSmaPA8q4jtYxv-BUnWMbNwTDl_0PcQovCyyCCczqm2ByB5X4E8_BevMbsdfdOxtQf9yFrxnST0aS3rYuQ8rxXFW5XYsJb_X5BsLegggA9LI6kxuYtd_BO-pJarpsO0yhWmiPZ4dkU-7ziCwTM5DxsGCFK61mpLGmM3gb25cuCaz1UpnboRpZhXbKHiTLfm-6XL_sYDum2fuohiYeg4gLv7nkY198iiCejKhPVKcZ5wgN0CMZ3JmsgbwELyY1dGfOxNeeX4Z0tg.hyHM-Hzr4VkeEiFh8IEEcA/__results___files/__results___14_0.png)

#### **CLASS 4 : NO BIRD**
- Given the amount of background noise/clutter (non bird pixels) we have in our images, <b>no_bird images seem like significant additions</b>
- Especially important are the <b>flag</b> & <b>feeder</b> images which we saw in the background of some of the hummingbird images.
- No birds are present in any of the images, showing the environment around the feeder

![](https://www.kaggleusercontent.com/kf/99926830/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nnCBGmtM_lkYAJo4IIyhLg.McJxw2Cq5L0j1Awa_cuPV3Psv6rfmJugWJ7DSYJ14sWyAMnjmJ8KHNXqhn3AXcKWvScQLPRnw5P4o1hpdhaPii5_EFU3AYrA_MLHkwXPMUkShmCwxX6yJ8MhDmiMTOYw281AbJMj4-ErMSlrC4aX6UMuyHiGFG4Kdp1itfrVzL0L23GH2gs8g2xSKTUkw8jeJepReHRxvT8_z1TIPOznnytVSP9odtaaD8P_3G_-3NywZIPgVoBQdgm9cHIhgPDFois-IiWXKFxEmhWL5gsdu4UViFACNoBkhxh0IzSnvLqnBWTVcY6sTX8Ta1fEfa_SnlFMNk2ijBEJEzlhQQwOKew8kwlCH_tLNdn9m8U9q_9plL_BaI_scpdEVI0FQUPAhf7m6aYVXy2LZaC_GBaIXWsJTN0ZuGk2PdT57K3Q415ljMRx5ILBkmU9utbjGx3Ft8RVw_FKkxyhSmaPA8q4jtYxv-BUnWMbNwTDl_0PcQovCyyCCczqm2ByB5X4E8_BevMbsdfdOxtQf9yFrxnST0aS3rYuQ8rxXFW5XYsJb_X5BsLegggA9LI6kxuYtd_BO-pJarpsO0yhWmiPZ4dkU-7ziCwTM5DxsGCFK61mpLGmM3gb25cuCaz1UpnboRpZhXbKHiTLfm-6XL_sYDum2fuohiYeg4gLv7nkY198iiCejKhPVKcZ5wgN0CMZ3JmsgbwELyY1dGfOxNeeX4Z0tg.hyHM-Hzr4VkeEiFh8IEEcA/__results___files/__results___16_0.png)

#### **REVIEWING THE DATA**
- In the context of bird monitoring, what I think this dataset outlines more than anything else is that:
  - You don't need to place cameras right next to the feeder (for some species can be offputting
  - The images don't need to be of perfect quality 

- Most hummingbirds are very similar in shape and are <b>mostly differentiable by their colours</b>:
  - So 1 channel CNN input network would be less effective, compared to a 3 channel network, and we have to rely on all colour channels to distinguish the species.
- Having gone through the images, we can see that the current dataset is quite a challenging one. A lot of other hummingbirds, especially male have very <b>identifiable feather colours</b>, however in this dataset, aside from the <b>broadtail male</b>, broadtail and rufus female hummingbirds <b>seem amost identical to the naked eye</b>.

### 7 | Baseline model

#### CREATE DATA GENERATORS

- Before creating <code>data generators</code>, we set set the image augmentation options via <code>ImageDataGenerator</code>, 
- For the **baseline** model, we'll utilise standard scaling <code>rescale</code>
- Let's create <code>data generators</code>, our data are in separate folders so we'll use <code>flow_from_directory</code>
- We'll split the entire dataset into groups & create data of <code>batch size</code> 32 & resize the images to **(100,100)** px containing **3 channels**

```python

# Define DataGenerators
train_datagen = ImageDataGenerator(rescale=1.0/255)
gen_datagen = ImageDataGenerator(rescale=1.0/255)

# DataGenerators via Folder Directory
gen_train = train_datagen.flow_from_directory(train_folder, 
                        target_size=(cfg.sshape[0],cfg.sshape[1]),  # target size
                        batch_size=32,                              # batch size
                        class_mode='categorical')    

gen_valid = gen_datagen.flow_from_directory(val_folder,
                        target_size=(cfg.sshape[0],cfg.sshape[1]),  # target size
                        batch_size=32,                              # batch size
                        class_mode='categorical')

gen_test = gen_datagen.flow_from_directory(test_folder,
                        target_size=(cfg.sshape[0],cfg.sshape[1]),  # target size
                        batch_size=32,                              # batch size
                        class_mode='categorical')
```

```
Found 400 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
```

#### DEFINE CONVOLUTED NEURAL NETWORK
- Let's define a neural network:
  - two <code>Conv2D</code> layers
  - a maxpooling layer (default (2,2)) pool size)
  - a <code>flattern</code> layer, which creates a single dimension tensor
  - <code>dropout</code> & <code>dense</code> layers interchangably
  - The final <code>dense</code> layer containing a <code>softmax</code> activation function

```python

# Two Convolution Layer CNN
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=3,
                            padding="same", 
                            activation="relu", 
                            input_shape=cfg.sshape),    
    keras.layers.Conv2D(64, kernel_size=3, 
                            padding="same", 
                            activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(cfg.labels, activation="softmax")
])

# Show the Model Architecture
model.summary()
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 100, 100, 32)      896       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 64)      18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 50, 50, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 160000)            0         
_________________________________________________________________
dropout (Dropout)            (None, 160000)            0         
_________________________________________________________________
dense (Dense)                (None, 128)               20480128  
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 516       
=================================================================
Total params: 20,500,036
Trainable params: 20,500,036
Non-trainable params: 0
_________________________________________________________________
```

#### COMPILE MODEL

- For evauation metrics:
  - We can use the inbuilt ones (callable via strings eg. 'acc')
  - Or reference to a function, let's define functions for <code>recall</code>,<code>precision</code>,<code>f1</code>

```python

# Evaluation Metrics for Callback
def get_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def get_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def get_f1(y_true, y_pred):
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

```

- Compile the model:
  - With an <code>optimiser</code> set to **Adam**
  - <code>loss</code> function set to **categorical_crossentropy** (multiclass classification tasks)

```python
''' Model Compilation '''
model.compile(optimizer='Adam', 
              loss='categorical_crossentropy',
              metrics=['acc',get_f1,get_precision,get_recall])
```

#### TRAIN MODEL

- Set <code>callbacks</code>:
  - <code>ReduceLROnPlateau</code>, which adjusts the learning data during training
  - <code>ModelCheckpoint</code>, which saves model data during training

```python
''' Callback Options During Training '''
callbacks = [ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=0, 
                               factor=0.5,mode='max',min_lr=0.001),
             ModelCheckpoint(filepath=f'model_cnn.h5',monitor='val_accuracy',
                             mode = 'max',verbose=0,save_best_only=True),
             TqdmCallback(verbose=0)] 
```

- Traing the model, using the <code>fit</code> method for 50 <code>iterations</code>

```python

''' Start Training '''
start = time.time()
history = model.fit(gen_train,
                    validation_data = gen_valid,
                    callbacks=callbacks,
                    verbose=0,
                    epochs=cfg.n_epochs  # Training for n_epoch interations
                   )
end = time.time()
print(f'The time taken to execute is {round(end-start,2)} seconds.')
print(f'Maximum Train/Val {max(history.history["acc"]):.4f}/{max(history.history["val_acc"]):.4f}')
```

```
50/50 [00:52<00:00, 1.03epoch/s, loss=0.000977, acc=1, get_f1=1, get_precision=1, 
      get_recall=1, val_loss=1.92, val_acc=0.762, val_get_f1=0.781, val_get_precision=0.781, val_get_recall=0.781, lr=0.001]
```

#### PLOT KERAS HISTORY

- To plot keras' <code>history</code> output, we can use a custom function <code>plot_keras_metric</code>

```python

# Function to plot loss & metric (metric_id)
def plot_keras_metric(history):

    # Palettes
    lst_color = ['#B1D784','#2E8486','#004379','#032B52','#EAEA8A']
    metric_id = ['loss','get_f1']
    fig = make_subplots(rows=1, cols=len(metric_id),subplot_titles=metric_id)

    jj=0;
    for metric in metric_id:     

        jj+=1

        # Main Trace
        fig.add_trace(go.Scatter(x=[i for i in range(1,cfg.n_epochs+1)],
                                 y=history.history[metric],
                                 name=f'train_{metric}',
                                 line=dict(color=lst_color[0]),mode='lines'),
                      row=1,col=jj)
        fig.add_trace(go.Scatter(x=[i for i in range(1,cfg.n_epochs+1)],
                                 y=history.history['val_'+metric],
                                 name=f'valid_{metric}',
                                 line=dict(color=lst_color[3]),mode='lines'),
                      row=1,col=jj)

    fig.update_layout(yaxis=dict(range=[0,1]),yaxis_range=[0,1],
                      height=400,width=650,showlegend=False,template='plotly_white',
                      hovermode="x",title=f'Training')
    
    fig['layout']['yaxis'].update(title='', range=[0,5], autorange=True,type='log')
    fig['layout']['yaxis2'].update(title='', range=[0, 1.1], autorange=False)
    fig.show()
```

### 8 | Image Agumentation Models

#### CREATE A CUSTOM TRAINING FUNCTION
- Let's create a helper function, that will input a <code>list</code> of <code>ImageDataGenerators</code>, containing the relevant image data augmentations that we want to apply to the dataset
- <code>model</code>, <code>compilation</code> settings are unchanged from the **baseline model**

```python

''' Evaluate CNN model w/ imported list of augmentation options '''
# augment_model inputs nested lists of augmentation options & evaluates 

def augment_model(lst_aug):

    # Define DataGenerator, load image data from directory
    gen_train = lst_aug[0].flow_from_directory(train_folder, 
                            target_size=(cfg.sshape[0],cfg.sshape[1]),  # target size
                            batch_size=32,          # batch size
                            class_mode='categorical')    # batch size

    gen_valid = lst_aug[1].flow_from_directory(val_folder,
                            target_size=(cfg.sshape[0],cfg.sshape[1]),
                            batch_size=32,
                            class_mode='categorical')

    gen_test = lst_aug[1].flow_from_directory(test_folder,
                            target_size=(cfg.sshape[0],cfg.sshape[1]),
                            batch_size=32,
                            class_mode='categorical')

    # Define a CNN Model
    model = keras.models.Sequential([
        
        keras.layers.Conv2D(32, kernel_size=3, 
                                padding="same", 
                                activation="relu", 
                                input_shape=cfg.sshape),    
        keras.layers.Conv2D(64, 
                            kernel_size=3, 
                            padding="same", 
                            activation="relu"),
        
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(cfg.labels, activation="softmax")
    ])
    
    # Compile Model
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['acc',get_f1,get_precision,get_recall])
    
    # Callback Options During Training 
    callbacks = [ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=0, 
                                   factor=0.5,mode='max',min_lr=0.001),
                 ModelCheckpoint(filepath=f'model_out.h5',monitor='val_accuracy',
                                 mode = 'max',verbose=0,save_best_only=True),
                 TqdmCallback(verbose=0)] 
    
    # Evaluate Model
    history = model.fit(gen_train,
                        validation_data = gen_valid,
                        callbacks=callbacks,
                        verbose=0,epochs=cfg.n_epochs)
    
    # Return Result History
    return history 

```

#### HELPER FUNCTIONS FOR TRAINING LOOP

- Define helper function <code>get_aug_name</code>, for main training loop <code>aug_eval</code>
- The desired augmentations are stored in a <code>list</code> (numerical form, corresponding to the index found in <code>lst_augopt</code>)
- This is done to reduce the input size, as the names can get quite long, so its just for convenience
- <code>lst_augval</code> contains the relevant augmentation values

```python

# lst of augmentation options
lst_augopt = ['rescale','horizontal_flip','vertical_flip',
              'brightness_range','rotation_range','shear_range',
              'zoom_range','width_shift_range','height_shift_range',
              'channel_shift_range','zca_whitening','featurewise_center',
              'samplewise_center','featurewise_std_normalization','samplewise_std_normalization']

# lst of default setting corresponding to lst_augopt
lst_augval = [1.0/255,True,True,  
              [1.1,1.5],0.2,0.2,
              0.2,0,0,
              0,True,False,
             False,False,False]

# Get Augmentation Names from lst_select options
def get_aug_name(lst_select):
    lst_selectn = [];
    for i in lst_select:
        tlst_all = []
        for j in i:
            tlist_selectn = tlst_all.append(lst_augopt[j])
        lst_selectn.append(tlst_all)
    return lst_selectn

```

#### DEFINE TRAINING FUNCTION

- We define a training function, which will loop through all given combinations of augmentation, <code>lst_select</code>
- Generate <code>DataGenerators</code> that correpond to the particular augmentation & train the model, saving the <code>history</code> for each case

```python

# Model Evaluation w/ Augmentation
def aug_eval(lst_select=None):

    ii=-1; lst_history = []
    for augs in lst_select:

        print('Augmentation Combination')
        # get dictionary of augmentation options
        ii+=1; dic_select = dict(zip([lst_augopt[i] for i in lst_select[ii]],[lst_augval[i] for i in lst_select[ii]]))
        print(dic_select)

        # define augmentation options
        train_datagen = ImageDataGenerator(**dic_select) # pass arguments
        gen_datagen = ImageDataGenerator(rescale=1.0/255)

        # evaluate model & return history metric
        history = augment_model([train_datagen,gen_datagen])

        # store results
        lst_history.append(history)
```        

#### AUGMENTATION COMBINATIONS

Let's test four different image <code>augmentation</code> combinations:
- <b>Combination 1</b> : <code>rescale (1/255)</code>, <code>horizontal_flip</code>
- <b>Combination 2</b> : <code>rescale (1/255)</code>, <code>vertical_flip</code>
- <b>Combination 3</b> : <code>rescale (1/255)</code>, <code>brightness_range (+1.1,+1.5)</code>
- <b>Combination 4</b> : <code>rescale (1/255)</code>, <code>horizontal_flip</code>, <code>shear_range (0.2)</code>, <code>zoom_range (0.2)</code>


```python

# Select Augmentation
lst_select = [[0,1],[0,2],[0,3],[0,1,5,6]] # list of augmentations
lst_selectn = get_aug_name(lst_select)     # get list of augmentation names
print(lst_selectn)
```

```
[['rescale', 'horizontal_flip'], ['rescale', 'vertical_flip'], ['rescale', 'brightness_range'], ['rescale', 'horizontal_flip', 'shear_range', 'zoom_range']]
```

#### TRAIN THE MODELS
- Train the same model defined as before, on augment adjusted datasets
- The output of <code>aug_eval</code> will return as list of history results

```python
lst_select = [[0,1],[0,2],[0,3],[0,1,5,6]] # list of augmentations
history = aug_eval(lst_select)             # get list of history results
```

```
Augmentation Combination
{'rescale': 0.00392156862745098, 'horizontal_flip': True}
Found 400 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
50/50 [00:46<00:00, 1.01epoch/s, loss=0.0237, acc=0.995, get_f1=0.996, get_precision=0.998,
       get_recall=0.995, val_loss=1.19, val_acc=0.775, val_get_f1=0.777, val_get_precision=0.794, val_get_recall=0.76, lr=0.001]
```

```
Augmentation Combination
{'rescale': 0.00392156862745098, 'vertical_flip': True}
Found 400 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
50/50 [00:47<00:00, 1.10epoch/s, loss=0.011, acc=0.998, get_f1=0.998, get_precision=0.998,
       get_recall=0.998, val_loss=1.11, val_acc=0.788, val_get_f1=0.781, val_get_precision=0.781, val_get_recall=0.781, lr=0.001]
```

```
Augmentation Combination
{'rescale': 0.00392156862745098, 'brightness_range': [1.1, 1.5]}
Found 400 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
50/50 [00:54<00:00, 1.18s/epoch, loss=0.114, acc=0.96, get_f1=0.96, get_precision=0.963, 
       get_recall=0.957, val_loss=1.33, val_acc=0.725, val_get_f1=0.673, val_get_precision=0.748, val_get_recall=0.615, lr=0.001]
```

```
Augmentation Combination
{'rescale': 0.00392156862745098, 'horizontal_flip': True, 'shear_range': 0.2, 'zoom_range': 0.2}
Found 400 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
Found 80 images belonging to 4 classes.
50/50 [01:26<00:00, 1.66s/epoch, loss=0.15, acc=0.947, get_f1=0.95, get_precision=0.959, 
       get_recall=0.942, val_loss=0.612, val_acc=0.85, val_get_f1=0.841, val_get_precision=0.85, val_get_recall=0.833, lr=0.001]
```

#### REMARKS
- **Brightness Augmentation**
  - Due to the low brightness nature of a lot of images, an increase in brightness would allowed the model to more easily distinguish between different classes
  - We can see that when just by the applying the increased brightness augmentation (<b>Combination 3</b>); [0,3]  set to (+1.1,+1.5), the model outperforms all other variations within the first 5 iterations, both on <b>training</b> & <b>validation</b> datasets, after which the validation accuracy starts to stagnate, and the model starts to show signs of overfitting
  
- **Other Observations**
  - What was interesting to observe was the <b>balance between training/validation accuracies</b>
  - Models with lots of augmentation combinations (<b>Combination 4</b>) tended to learned slower, ended up with lower training accuracies but generalised better on unseen data
  - Simple Horizontal flipping, [0,1] (<b>Combination 1</b>) and the combination of four augmentations (shearing,zooming,flipping) [0,1,5,6], both were more effective than simply applying a brightness augmentation adjustments [0,3]

### 9 | Transfer Learning Models

#### DEFINE DATALOADERS

- Based on the results from the previous section (Image Agumentation Models):
  - We'll be using the most successful combination for augmentation, which was **combination 4**
- The augmentations for the training <code>datagenerator</code>: <code>shear_range</code>, <code>zoom_range</code> & <code>horizontal_flip</code>

```python

# Using the best augmentation combination
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                  )
gen_datagen = ImageDataGenerator(rescale=1.0/255)

gen_train = train_datagen.flow_from_directory(train_folder, 
                        target_size=(cfg.sshape[0],cfg.sshape[1]),  # target size
                        batch_size=32,          # batch size
                        class_mode='categorical')    # batch size

gen_valid = gen_datagen.flow_from_directory(val_folder,
                        target_size=(cfg.sshape[0],cfg.sshape[1]),
                        batch_size=32,
                        class_mode='categorical')

gen_test = gen_datagen.flow_from_directory(test_folder,
                        target_size=(cfg.sshape[0],cfg.sshape[1]),
                        batch_size=32,
                        class_mode='categorical')

```

#### ASSEMBLE PRETRAINED CLASSIFIER MODEL 

- Let's look at four different models <code>VGG</code>, <code>ResNet</code>, <code>MobileNet</code>, <code>InceptionV3</code> & <code>EfficientNetB4</code>
- We'll split the model into **two parts** <code>head</code> & <code>tail</code> then assemble a new <code>sequential</code> model:
  - The head part will contain the pretrained model (eg. the VGG model)
  - The tail end will contain our classifier part (<code>flatten</code>, <code>dense</code> & <code>dropout</code> layers)

```python

# from tensorflow.keras import applications as app
def pretrained_model(head_id):

    # Define model with different applications
    model = Sequential()

    ''' Define Head Pretrained Models '''

    if(head_id is 'vgg'):
        model.add(app.VGG16(input_shape=cfg.sshape,
                            pooling='avg',
                            classes=1000,
                            include_top=False,
                            weights='imagenet'))

    elif(head_id is 'resnet'):
        model.add(app.ResNet101(include_top=False,
                               input_tensor=None,
                               input_shape=cfg.sshape,
                               pooling='avg',
                               classes=100,
                               weights='imagenet'))

    elif(head_id is 'mobilenet'):
        model.add(app.MobileNet(alpha=1.0,
                               depth_multiplier=1,
                               dropout=0.001,
                               include_top=False,
                               weights="imagenet",
                               input_tensor=None,
                               input_shape = cfg.sshape,
                               pooling=None,
                               classes=1000))

    elif(head_id is 'inception'):
        model.add(InceptionV3(input_shape = cfg.sshape, 
                                 include_top = False, 
                                 weights = 'imagenet'))

    elif(head_id is 'efficientnet'):
        model.add(EfficientNetB4(input_shape = cfg.sshape, 
                                    include_top = False, 
                                    weights = 'imagenet'))

    ''' Tail Model Part '''
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(cfg.labels,activation='softmax'))

    # # freeze main model coefficients
    model.layers[0].trainable = False
    model.compile(optimizer='Adam', 
                  loss='categorical_crossentropy',
                  metrics=['acc',get_f1,get_precision,get_recall])

    return model # return compiled model

```

#### DEFINE TRAINING FUNCTION

- In this case, we will be looping through all the pretrained <code>head</code> models, as some of them showed better metric performance, compared to the base model

```python
# Pretrained Loaded Model 
def pretrain_eval(lst_heads,verbose=False):

    lst_history = []
    for head_id in lst_heads:

        # Define CNN Model
        model = pretrained_model(head_id)

        ''' Callback Options During Training '''
        callbacks = [ReduceLROnPlateau(monitor='val_acc',patience=2,verbose=0, 
                                       factor=0.5,mode='max',min_lr=0.001),
                     ModelCheckpoint(filepath=f'model_{head_id}.h5',monitor='val_acc',
                                     mode = 'max',verbose=0,save_best_only=True),
                     TqdmCallback(verbose=0)] 

        ''' Start Training '''
        start = time.time()
        history = model.fit(gen_train,
                            validation_data = gen_valid,
                            callbacks=callbacks,
                            verbose=0,
                            epochs=cfg.n_epochs)
        end = time.time()
        if(verbose):
            print(f'Head Model: {head_id}')
            print(f'The time taken to execute is {round(end-start,2)} seconds.')
            print(f'Maximum Train/Val {max(history.history["acc"]):.4f}/{max(history.history["val_acc"]):.4f}')

        lst_history.append(history)
        
    return lst_history

```

#### TRAIN THE MODEL

```
''' Define Model Architectre '''
lst_heads = ['vgg','resnet','mobilenet','inception','efficientnet']
history = pretrain_eval(lst_heads)
```

### 10 | Model Inference

- As the pretrained models quite substantially outperformed the CNN model we defined in **Section 7**
- Let's check how well these models perform on unseen test data

```python

for head_id in lst_heads:
    print(f'Head Model: {head_id}')
    
    # Load uncompiled model
    load_model = models.load_model(f"model_{head_id}.h5",compile=False)
    
    # Compile model
    load_model.compile(optimizer='Adam',
                       loss='categorical_crossentropy',
                       metrics=['acc',get_f1,get_precision,get_recall])
    
    # Evaluate on test dataset
    scores = load_model.evaluate(gen_test, verbose=1)

```

```

Head Model: vgg
3/3 [==============================] - 1s 127ms/step - loss: 0.3289 - acc: 0.8992 - get_f1: 0.9010 - get_precision: 0.9275 - get_recall: 0.8763
Head Model: resnet
3/3 [==============================] - 3s 47ms/step - loss: 1.0242 - acc: 0.5891 - get_f1: 0.3449 - get_precision: 0.7534 - get_recall: 0.2266
Head Model: mobilenet
3/3 [==============================] - 1s 38ms/step - loss: 0.4240 - acc: 0.8867 - get_f1: 0.8721 - get_precision: 0.8799 - get_recall: 0.8646
Head Model: inception
3/3 [==============================] - 2s 48ms/step - loss: 0.4712 - acc: 0.7750 - get_f1: 0.7865 - get_precision: 0.8108 - get_recall: 0.7643
Head Model: efficientnet
3/3 [==============================] - 5s 46ms/step - loss: 13.1678 - acc: 0.2539 - get_f1: 0.2487 - get_precision: 0.2487 - get_recall: 0.2487

```
