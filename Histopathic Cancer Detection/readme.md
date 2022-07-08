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

![](https://www.kaggleusercontent.com/kf/100338295/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..uOM96h0EZBWxG4rzjxQYDA.BrgFx4z8WelP0ZUx_4qQKBArv5LditubwOFuig0pJMEfaPrF879k8iBqvIlFhXDDFosqsTPQZUJSOx6sCJ8YbtMooybd2iXGUxrj2p8lZt0oU2NzY7a9Y3pMWNg076I1uBKGYzkF2iq69glpDoQNH433QBx2xCpb4qNWYajFJCzNyPaM-wTCcD6HUIx2irGhchoUTy4_zpHfXJ25PNkTdrm6seagFqH9iMTJ3u9i9znBe-K_qf7VsjeQN1sNylVPQXU3zsZ8mksViCvtiP3gdIIFeHuV3lOHgvffIRjFWoMiSNISo6aJeYD4wLIAv-htU0YRzkbm6bi0fVO83nHYoErayoPUQKwxR0gsZAHFCn0b9IenQlJwlOqh5M1OSQVaDCVpfREPr3L_fnD7Tf4aO1vOzGKTghVPvwP9qgQGqZ1jbJMnmSL6aYpF3Z5kG1qvI2FFePPiHBYqMktKbVYEIvl4-kzrGXzzau_OhWbjMc3G83kJVk1FTX0kOnrlnMimmUr5oDah3x-ZRFy_J9hOh3FjVqwQp7QmRDTjBSFKBkb-rhxYvWEaFPHm21v2d64nRzS2zIBMooSMA_TmOfi1zxdi0JxfddgIZd59ioOZaHK_M_8-BAMbYtmR4yU0o9hN-Nj0S_btnZcGbWi1DU0GdDB1pNgNc8FhQGSOvWN_x6cfsxQTH5vLD0urACm-r9Kb.TFfIelyYMMAgF_FM6juENQ/__results___files/__results___17_0.png)

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

![](https://www.kaggleusercontent.com/kf/100338295/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..uOM96h0EZBWxG4rzjxQYDA.BrgFx4z8WelP0ZUx_4qQKBArv5LditubwOFuig0pJMEfaPrF879k8iBqvIlFhXDDFosqsTPQZUJSOx6sCJ8YbtMooybd2iXGUxrj2p8lZt0oU2NzY7a9Y3pMWNg076I1uBKGYzkF2iq69glpDoQNH433QBx2xCpb4qNWYajFJCzNyPaM-wTCcD6HUIx2irGhchoUTy4_zpHfXJ25PNkTdrm6seagFqH9iMTJ3u9i9znBe-K_qf7VsjeQN1sNylVPQXU3zsZ8mksViCvtiP3gdIIFeHuV3lOHgvffIRjFWoMiSNISo6aJeYD4wLIAv-htU0YRzkbm6bi0fVO83nHYoErayoPUQKwxR0gsZAHFCn0b9IenQlJwlOqh5M1OSQVaDCVpfREPr3L_fnD7Tf4aO1vOzGKTghVPvwP9qgQGqZ1jbJMnmSL6aYpF3Z5kG1qvI2FFePPiHBYqMktKbVYEIvl4-kzrGXzzau_OhWbjMc3G83kJVk1FTX0kOnrlnMimmUr5oDah3x-ZRFy_J9hOh3FjVqwQp7QmRDTjBSFKBkb-rhxYvWEaFPHm21v2d64nRzS2zIBMooSMA_TmOfi1zxdi0JxfddgIZd59ioOZaHK_M_8-BAMbYtmR4yU0o9hN-Nj0S_btnZcGbWi1DU0GdDB1pNgNc8FhQGSOvWN_x6cfsxQTH5vLD0urACm-r9Kb.TFfIelyYMMAgF_FM6juENQ/__results___files/__results___18_0.png)
    
