![](https://i.imgur.com/0oW3bQA.png)

Project available on Kaggle: **[PyTorch | CNN Binary Image Classification](https://www.kaggle.com/code/shtrausslearning/hummingbird-classification-keras-cnn-models)** <br>
Dataset available on Kaggle: **[Histopathologic Cancer Detection]([https://www.kaggle.com/datasets/akimball002/hummingbirds-at-my-feeders](https://www.kaggle.com/competitions/histopathologic-cancer-detection))** <br>

Project Keywords: <br>
<code>keras</code> <code>CNN</code> <code>multiclass</code> <code>classification</code> <code>augmentation</code> <code>dataset from folder</code> <code>pretrained</code> <code>inference</code>

### 1 | Introduction

- Microscopic evaluation of histopathalogic stained tissue & its **subsequent digitalisation** is now a more feasible due to the advances in slide scanning technology, as well a reduction in digital storage cost in recent years
- There are certain advantages that come with such **digitalised pathology**; including remote diagnosis, instant archival access & simplified procedure of consultations with expert pathologists


- Examples of digitalised histopathalogic stained tissues:

![](https://i.imgur.com/9CguKyI.png)


- Digitalised Analysis based on <b>Deep Learning</b> has shown potential benefits as a <b>potential diagnosis tool</b> & strategy 
- Examples from literature: [Gulshan et al](https://jamanetwork.com/journals/jama/fullarticle/2588763) | [Esteva et al](https://pubmed.ncbi.nlm.nih.gov/28117445/) 
- Both these papers demonstrated the <b>potential of deep learning for diabetic retinopathy screening</b> and <b>skin lesion classification</b>, respectively
- An essential task performed by pathologist; **accurate breast cancer staging**
- Assessment of the extent of cancer spread by **histopathological analysis** of sentinel axillary lymph nodes (SLNs) is an essential part of breast cancer staging process

### 2 | Problem Statement

- The sensitivity of SLN assessment by pathologists, is not optimal
- A retrospective study showed that pathology review by experts changed the nodal status in 24% of patients
- SLN assessment is also <b>tedious</b> and <b>time-consuming</b>
- It has been shown that **Deep Learning** (DL) algorithms could identify metastases in SLN slides with 100% sensitivity, whereas 40% of the slides without metastases could be identified as such
- This could result in a <b>significant reduction in the workload</b> of pathologists

### 3 | Study Aim

- The aim of this study is to investigate the potential of using <code>Pytorch</code> Deep Learning module for the <b>detection of metastases</b> in SLN slides and compare them with the predefined pathologist diagnosis (expert evaluations)
