![](https://i.imgur.com/XilezGZ.png)

## Aim

- The aim of this project is to create a model that will be able to determine from <code>x-rays</code>, which patiets have been labeled corona positive (option 1) and which are standard x-rays (option 2)
- The model will need to find subtle patterns in the images, so we'll need to utilise <code>CNN</code> models & build a <code>binary classifier</code> 
- We will need to find both normal and corona positive <code>x-ray</code> images & utilise the <code>keras</code> deep learning module to classify images 

## Keywords

<code>x-rays images</code> <code>self sorted folders</code> <code>keras</code> <code>CNN model</code> <code>flow_from_directory</code> <code>image augmentation</code>

## Dataset
The dataset being used in this project have been combined from two different sources **[source 1](https://github.com/ieee8023/covid-chestxray-dataset)** **[source 2](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**
- <code>Source 1</code> contains images from <code>x-rays</code> of patients who have been labeled to be corona positive patients
- <code>Source 2</code> contains images from normal <code>x-rays</code>

Image <code>resolution</code>:
- (224,224,3) 

The following <code>augmentations</code> were used:
- shear_range=0.2
- zoom_range=0.2
- horizontal_flip=True
- rescale=1/255


## Files
<code>main.py</code> - Training File

## Process
- [1] First, we need to create a root folder, in which we will be placing our dataset
- [2] We need to then create a folder, which will contain both <code>train</code> & <code>validation</code> folders
- [3] For both <code>train</code> & <code>validation</code>, we need to decide a distribution (how much data is used in training & validation)
- [4] Once decided, <code>x-rays</code> will be sorted into two folders <code>covid</code> & <code>normal</code>
- [5] Run <code>main.py</code>, which will display the <code>accuracy</code> on both <code>training</code> & </code>validation</code> datasets

## Result
- After 10 <code>epoch</code> (dataset passes), the <code>CNN</code> model accuracy for training was **0.9480** and **0.9667**

