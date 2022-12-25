
![](https://i.imgur.com/0oW3bQA.png)

![](https://camo.githubusercontent.com/d38e6cc39779250a2835bf8ed3a72d10dbe3b05fa6527baa3f6f1e8e8bd056bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f64652d507974686f6e2d696e666f726d6174696f6e616c3f7374796c653d666c6174266c6f676f3d707974686f6e266c6f676f436f6c6f723d776869746526636f6c6f723d326262633861) ![](https://badgen.net/badge/status/organising/blue) 

### Project Keywords 📒

`binary` `classification` `breast` `cancer` `histopathalogic` `analysis` `SLN` `PyTorch`

### Project Background 📥

- Microscopic evaluation of histopathalogic stained tissue & its subsequent digitalisation is now a more feasible due to the advances in slide scanning technology, as well a reduction in digital storage cost in recent years
- There are certain advantages that come with such digitalised pathology; including remote diagnosis, instant archival access & simplified procedure of consultations with expert pathologists
- Digitalised Analysis based on Deep Learning has shown potential benefits as a potential diagnosis tool & strategy
- [Gulshan et al](https://jamanetwork.com/journals/jama/fullarticle/2588763) and [Esteva et al](https://pubmed.ncbi.nlm.nih.gov/28117445/) demonstrated the <b>potential of deep learning for diabetic retinopathy screening</b> and <b>skin lesion classification</b>, respectively
- An essential task performed by pathologist; accurate breast cancer staging 
- Assessment of the extent of **cancer spread by histopathological analysis** of sentinel axillary lymph nodes `SLN` is an **essential part of breast cancer staging process**
- The sensitivity of `SLN` assessment by pathologists, however, is not optimal. A retrospective study showed that **pathology review by experts** changed the nodal status in 24% of patients
- `SLN` assessment is <b>tedious</b> and <b>time-consuming</b>. It has been shown that DL algorithms could identify metastases in `SLN` slides with 100% sensitivity, whereas 40% of the slides without metastases could be identified as such
- This could result in a <b>significant reduction in the workload</b> of pathologists

### Project Aim 🎯 

- The aim of this study was to investigate the potential of using the `PyTorch` DL module for the <b>detection of metastases</b> in `SLN` slides and compare them with the predefined pathologist diagnosis
- The secondary aim of this study is to understand each component of the `PyTorch` model creation & inference process so it can be easily applied to other problems

### Kaggle Notebook 📖

- Kaggle offer a very neat `ipynb` render: **[PyTorch | CNN Binary Image Classification](https://www.kaggle.com/code/shtrausslearning/pytorch-cnn-binary-image-classification)**

### Project Pipeline 📑

Kaggle notebook workflow

- `1` Introduction
- `2` The Dataset
- `3` Data Preparation
- `4` Splitting the Dataset (Training,Vaidation)
- `5` Creating Dataloaders
- `6` Image Augmentation
- `7` Binary Classifier Model
- `8` Defining Loss Function 
- `9` Defining Optimiser
- `10` Training Model
- `11` Inference on Test Set

### Main Takeaways 📤

Main takeways for this study are related to `PyTorch` usage, we'll go through the project

- We start off by placing all the image data into two folders `train` & `test`
  - `Training` images are used to create training & validation **image subsets** (custom dataset class)

The custom `dataset` class requires:
- `__getitem__` (returns tensor,label for specific index)
- `__len__` (size of data)

<br>

```python

class pytorch_data(Dataset):
    
    def __init__(self,data_dir,transform,data_type="train"):      
    
        # Get Image File Names
        cdm_data=os.path.join(data_dir,data_type)  # directory of files
        
        file_names = os.listdir(cdm_data) # get list of images in that directory  
        idx_choose = np.random.choice(np.arange(len(file_names)), 
                                      4000,
                                      replace=False).tolist()
        file_names_sample = [file_names[x] for x in idx_choose]
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names_sample]   # get the full path to images
        
        # Get Labels
        labels_data=os.path.join(data_dir,"train_labels.csv") 
        labels_df=pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True) # set data frame index to id
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names_sample]  # obtained labels from df
        self.transform = transform
      
    def __len__(self):
        return len(self.full_filenames) # size of dataset
      
    # open image, apply transforms and return with label
    def __getitem__(self, idx):
        
        # Open Image with PIL
        image = Image.open(self.full_filenames[idx])  
        
        # Apply Specific Transformation to Image to get tensor
        image = self.transform(image) 
        
        return image, self.labels[idx]
        
```

