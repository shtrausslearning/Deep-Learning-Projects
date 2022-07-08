![](https://i.imgur.com/0oW3bQA.png)

Project available on Kaggle: **[PyTorch | CNN Binary Image Classification](https://www.kaggle.com/code/shtrausslearning/hummingbird-classification-keras-cnn-models)** <br>
Dataset available on Kaggle: **[Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection)** <br>

Project Keywords: <br>
<code>PyTorch</code> <code>CNN</code> <code>binary</code> <code>classification</code> <code>image</code> <code>augmentation</code> <code>inference</code>

### 1 | Introduction

- Microscopic evaluation of histopathalogic stained tissue & its **subsequent digitalisation** is now a more feasible due to the advances in slide scanning technology, as well a reduction in digital storage cost in recent years
- There are certain advantages that come with such **digitalised pathology**; including remote diagnosis, instant archival access & simplified procedure of consultations with expert pathologists

- Examples of digitalised histopathalogic stained tissues:

![](https://i.imgur.com/9CguKyI.png)


- Digitalised Analysis based on <b>Deep Learning</b> has shown potential benefits as a <b>potential diagnosis tool</b> & strategy 
- Examples from literature: [Gulshan et al](https://jamanetwork.com/journals/jama/fullarticle/2588763) | [Esteva et al](https://pubmed.ncbi.nlm.nih.gov/28117445/) 
- Both these papers demonstrated the <b>potential of deep learning for diabetic retinopathy screening</b> and <b>skin lesion classification</b>, respectively
- An essential task performed by pathologist; **accurate breast cancer staging**
- Assessment of the extent of cancer spread by **histopathological analysis** of Sentinel axillary lymph nodes (**SLNs**) is an essential part of **breast cancer staging process**

### 2 | Problem Statement

- The sensitivity of SLN assessment by pathologists, is not optimal
- A retrospective study showed that pathology review by experts changed the nodal status in 24% of patients
- SLN assessment is also <b>tedious</b> and <b>time-consuming</b>
- It has been shown that **Deep Learning** (DL) algorithms could identify metastases in SLN slides with 100% sensitivity, whereas 40% of the slides without metastases could be identified as such
- This could result in a <b>significant reduction in the workload</b> of pathologists

### 3 | Study Aim

- The aim of this study is to investigate the potential of using <code>Pytorch</code> Deep Learning module for the <b>detection of metastases</b> in **SLN** slides and compare them with the predefined pathologist diagnosis (expert evaluations)

### 4 | Reading Data

- The dataset is split into two folders <code>training</code> & <code>test</code> sets
- The dataset contains an <code>csv</code> file, which contains the labels for all the images in both folders

```python
os.listdir('/kaggle/input/histopathologic-cancer-detection/')
```

```
['sample_submission.csv', 'train_labels.csv', 'test', 'train']
```

```python
labels_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
labels_df.head()
```

|    | id                                       |   label |
|---:|:-----------------------------------------|--------:|
|  0 | f38a6374c348f90b587e046aac6079959adf3835 |       0 |
|  1 | c18f2d887b7ae4f6742ee445113fa1aef383ed77 |       1 |
|  2 | 755db6279dae599ebb4d39a9123cce439965282d |       0 |
|  3 | bc3f0c64fb968ff4a8bd33af6971ecae77c75e08 |       0 |
|  4 | 068aba587a4950175d04c680d38943fd488d6a9d |       0 |

#### Target Class Distribution

- The <code>dataset</code> contains quite an evenly distributed class balance between <code>malignant</code> & <code>non-malingant</code> cases

```python
labels_df['label'].value_counts()
```

```
0    130908
1     89117
Name: label, dtype: int64
```

#### Malignant & Non-Malignant 

- Previewing samples from both datasets, we can see that it's quite difficult to distinguish between the two
- The <code>malignants</code> class images are outlined with **red boxes**
- The normal, <code>non-malignant</code> class images are outligned with **green boxes**

```python
imgpath ="/kaggle/input/histopathologic-cancer-detection/train/" # training data is stored in this folder
malignant = labels_df.loc[labels_df['label']==1]['id'].values    # get the ids of malignant cases
normal = labels_df.loc[labels_df['label']==0]['id'].values       # get the ids of the normal cases
```

```python
nrows,ncols=6,15
fig,ax = plt.subplots(nrows,ncols,figsize=(15,6))
plt.subplots_adjust(wspace=0, hspace=0) 
for i,j in enumerate(malignant[:nrows*ncols]):
    fname = os.path.join(imgpath ,j +'.tif')
    img = Image.open(fname)
    idcol = ImageDraw.Draw(img)
    idcol.rectangle(((0,0),(95,95)),outline='red')
    plt.subplot(nrows, ncols, i+1) 
    plt.imshow(np.array(img))
    plt.axis('off')
```

<img src="https://www.kaggleusercontent.com/kf/100338295/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..uOM96h0EZBWxG4rzjxQYDA.BrgFx4z8WelP0ZUx_4qQKBArv5LditubwOFuig0pJMEfaPrF879k8iBqvIlFhXDDFosqsTPQZUJSOx6sCJ8YbtMooybd2iXGUxrj2p8lZt0oU2NzY7a9Y3pMWNg076I1uBKGYzkF2iq69glpDoQNH433QBx2xCpb4qNWYajFJCzNyPaM-wTCcD6HUIx2irGhchoUTy4_zpHfXJ25PNkTdrm6seagFqH9iMTJ3u9i9znBe-K_qf7VsjeQN1sNylVPQXU3zsZ8mksViCvtiP3gdIIFeHuV3lOHgvffIRjFWoMiSNISo6aJeYD4wLIAv-htU0YRzkbm6bi0fVO83nHYoErayoPUQKwxR0gsZAHFCn0b9IenQlJwlOqh5M1OSQVaDCVpfREPr3L_fnD7Tf4aO1vOzGKTghVPvwP9qgQGqZ1jbJMnmSL6aYpF3Z5kG1qvI2FFePPiHBYqMktKbVYEIvl4-kzrGXzzau_OhWbjMc3G83kJVk1FTX0kOnrlnMimmUr5oDah3x-ZRFy_J9hOh3FjVqwQp7QmRDTjBSFKBkb-rhxYvWEaFPHm21v2d64nRzS2zIBMooSMA_TmOfi1zxdi0JxfddgIZd59ioOZaHK_M_8-BAMbYtmR4yU0o9hN-Nj0S_btnZcGbWi1DU0GdDB1pNgNc8FhQGSOvWN_x6cfsxQTH5vLD0urACm-r9Kb.TFfIelyYMMAgF_FM6juENQ/__results___files/__results___17_0.png" height="300">

```python

plt.rcParams['figure.figsize'] = (15, 6)
plt.subplots_adjust(wspace=0, hspace=0)

nrows,ncols=6,15
for i,j in enumerate(normal[:nrows*ncols]):
    fname = os.path.join(imgpath ,j +'.tif')
    img = Image.open(fname)
    idcol = ImageDraw.Draw(img)
    idcol.rectangle(((0,0),(95,95)),outline='green')
    plt.subplot(nrows, ncols, i+1) 
    plt.imshow(np.array(img))
    plt.axis('off')
```

<img src="https://www.kaggleusercontent.com/kf/100338295/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..uOM96h0EZBWxG4rzjxQYDA.BrgFx4z8WelP0ZUx_4qQKBArv5LditubwOFuig0pJMEfaPrF879k8iBqvIlFhXDDFosqsTPQZUJSOx6sCJ8YbtMooybd2iXGUxrj2p8lZt0oU2NzY7a9Y3pMWNg076I1uBKGYzkF2iq69glpDoQNH433QBx2xCpb4qNWYajFJCzNyPaM-wTCcD6HUIx2irGhchoUTy4_zpHfXJ25PNkTdrm6seagFqH9iMTJ3u9i9znBe-K_qf7VsjeQN1sNylVPQXU3zsZ8mksViCvtiP3gdIIFeHuV3lOHgvffIRjFWoMiSNISo6aJeYD4wLIAv-htU0YRzkbm6bi0fVO83nHYoErayoPUQKwxR0gsZAHFCn0b9IenQlJwlOqh5M1OSQVaDCVpfREPr3L_fnD7Tf4aO1vOzGKTghVPvwP9qgQGqZ1jbJMnmSL6aYpF3Z5kG1qvI2FFePPiHBYqMktKbVYEIvl4-kzrGXzzau_OhWbjMc3G83kJVk1FTX0kOnrlnMimmUr5oDah3x-ZRFy_J9hOh3FjVqwQp7QmRDTjBSFKBkb-rhxYvWEaFPHm21v2d64nRzS2zIBMooSMA_TmOfi1zxdi0JxfddgIZd59ioOZaHK_M_8-BAMbYtmR4yU0o9hN-Nj0S_btnZcGbWi1DU0GdDB1pNgNc8FhQGSOvWN_x6cfsxQTH5vLD0urACm-r9Kb.TFfIelyYMMAgF_FM6juENQ/__results___files/__results___18_0.png" height="300">
    
### 5 | Data Preparation (preview)
- The dataset contains two separate folders <code>train</code> & <code>test</code>
- The special data class, contains <code>__len__</code> & <code>__getitem__</code> special methods, returning the dataset size & (image & label) 

#### Custom dataset class

```python

torch.manual_seed(0) # fix random seed

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
      
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Open Image with PIL
        image = self.transform(image) # Apply Specific Transformation to Image
        return image, self.labels[idx]

```

#### Data transformations 

- Let's apply data transformations, by creating a <code>data_transformer</code> for the <code>training</code> data
- We'll use slightly smaller resolution (46,46,3) tensors, so let's preview how it will look like first

```python
# define transformation that converts a PIL image into PyTorch tensors
import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((46,46))])

# Define an object of the custom dataset for the train folder.
data_dir = '/kaggle/input/histopathologic-cancer-detection/'
img_dataset = pytorch_data(data_dir, data_transformer, "train") 

len_img=len(img_dataset)
len_train=int(0.8*len_img)
len_val=len_img-len_train

# Split Pytorch tensor
train_ts,val_ts=random_split(img_dataset,
                             [len_train,len_val]) # random split 80/20

print("train dataset size:", len(train_ts))
print("validation dataset size:", len(val_ts))
```

```
train dataset size: 3200
validation dataset size: 800
```

- Let's view the tensor data for every image & corresponding class <code>label</code>

```python
# getting the torch tensor image & target variable
ii=-1
for x,y in train_ts:
    print(x.shape,y)
    ii+=1
    if(ii>5):
        break
```

```
torch.Size([3, 46, 46]) 0
torch.Size([3, 46, 46]) 1
torch.Size([3, 46, 46]) 0
torch.Size([3, 46, 46]) 1
torch.Size([3, 46, 46]) 0
torch.Size([3, 46, 46]) 0
torch.Size([3, 46, 46]) 1
```

- Visualise the tensor data using <code>plotly</code>

```
import plotly.express as px

def plot_img(x,y,title=None):

    npimg = x.numpy() # convert tensor to numpy array
    npimg_tr=np.transpose(npimg, (1,2,0)) # Convert to H*W*C shape
    fig = px.imshow(npimg_tr)
    fig.update_layout(template='plotly_white')
    fig.update_layout(title=title,height=300,margin={'l':10,'r':20,'b':10})
    fig.show()
    
```

- Training data samples:

```python
# Create grid of sample images 
grid_size=30
rnd_inds=np.random.randint(0,len(train_ts),grid_size)
print("image indices:",rnd_inds)

x_grid_train=[train_ts[i][0] for i in rnd_inds]
y_grid_train=[train_ts[i][1] for i in rnd_inds]

x_grid_train=utils.make_grid(x_grid_train, nrow=10, padding=2)
print(x_grid_train.shape)
    
plot_img(x_grid_train,y_grid_train,'Training Subset Examples')
```

<img src="https://i.imgur.com/J0ge3nI.png" height="250">

- Validation data samples:

```python

grid_size=30
rnd_inds=np.random.randint(0,len(val_ts),grid_size)
print("image indices:",rnd_inds)
x_grid_val=[val_ts[i][0] for i in range(grid_size)]
y_grid_val=[val_ts[i][1] for i in range(grid_size)]

x_grid_val=utils.make_grid(x_grid_val, nrow=10, padding=2)
print(x_grid_val.shape)

plot_img(x_grid_val,y_grid_val,'Validation Dataset Preview')

```
<img src="https://i.imgur.com/mI4PrEw.png" height="250">
