
### Description

310 Observations, 13 Attributes (12 Numeric Predictors, 1 Binary Class Attribute - No Demographics)

Lower back pain can be caused by a variety of problems with any parts of the complex, interconnected network of spinal muscles, nerves, bones, discs or tendons in the lumbar spine. Typical sources of low back pain include:

- The large nerve roots in the low back that go to the legs may be irritated
- The smaller nerves that supply the low back may be irritated
- The large paired lower back muscles (erector spinae) may be strained
- The bones, ligaments or joints may be damaged
- An intervertebral disc may be degenerating
- An irritation or problem with any of these structures can cause lower back pain and/or pain that radiates or is referred to other parts of the body. Many lower back problems also cause back muscle spasms, which don't sound like much but can cause severe pain and disability.

While lower back pain is extremely common, the symptoms and severity of lower back pain vary greatly. A simple lower back muscle strain might be excruciating enough to necessitate an emergency room visit, while a degenerating disc might cause only mild, intermittent discomfort.

This data set is about to identify a person is <code>abnormal</code> or <code>normal</code> using collected physical spine details/data.


### Import Dataset

- Import dataset from **[dataset](https://www.kaggle.com/datasets/sammy123/lower-back-pain-symptoms-dataset)**, 
- <code>features</code> **12 features (anonimised)**, <code>targe</code> **Class_att**
- Original data contains <code>target</code> as <code>strings</code>, let's encode it with a <code>dictionary</code> encode_map

```python
import pandas as pd

df = pd.read_csv('../input/lower-back-pain-symptoms-dataset/Dataset_spine.csv')
df.drop([df.columns[-1]],axis=1,inplace=True)

df['Class_att'] = df['Class_att'].astype('category')
encode_map = {
    'Abnormal': 1,
    'Normal': 0
}

df['Class_att'].replace(encode_map, inplace=True)
display(df.head())
```

### Data Preparation

```python

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

y = df['Class_att']
X = df.drop(['Class_att'],axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.20,
                                                  random_state=13,
                                                  shuffle=True)
print(f'X_train: {X_train.shape}')
print(f'X_val: {X_val.shape}')

X_train = X_train.values
X_val = X_val.values
y_train = y_train.values
y_val = y_val.values

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

```
