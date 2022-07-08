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


```python
labels_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
labels_df.head()
```

```
'|    | id                                       |   label |\n|---:|:-----------------------------------------|--------:|\n|  0 | f38a6374c348f90b587e046aac6079959adf3835 |       0 |\n|  1 | c18f2d887b7ae4f6742ee445113fa1aef383ed77 |       1 |\n|  2 | 755db6279dae599ebb4d39a9123cce439965282d |       0 |\n|  3 | bc3f0c64fb968ff4a8bd33af6971ecae77c75e08 |       0 |\n|  4 | 068aba587a4950175d04c680d38943fd488d6a9d |       0 |'
```


```python
labels_df['label'].value_counts()
```

- The <code>dataset</code> contains 

```
0    130908
1     89117
Name: label, dtype: int64
```
