
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
- <code>features</code> **12 features (anonimised)**, <code>target</code> **Class_att**
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

|    |    Col1 |     Col2 |    Col3 |    Col4 |     Col5 |     Col6 |     Col7 |    Col8 |    Col9 |    Col10 |    Col11 |   Col12 |   Class_att |\n|---:|--------:|---------:|--------:|--------:|---------:|---------:|---------:|--------:|--------:|---------:|---------:|--------:|------------:|\n|  0 | 63.0278 | 22.5526  | 39.6091 | 40.4752 |  98.6729 | -0.2544  | 0.744503 | 12.5661 | 14.5386 | 15.3047  | -28.6585 | 43.5123 |           1 |\n|  1 | 39.057  | 10.061   | 25.0154 | 28.996  | 114.405  |  4.56426 | 0.415186 | 12.8874 | 17.5323 | 16.7849  | -25.5306 | 16.1102 |           1 |\n|  2 | 68.832  | 22.2185  | 50.0922 | 46.6135 | 105.985  | -3.53032 | 0.474889 | 26.8343 | 17.4861 | 16.659   | -29.0319 | 19.2221 |           1 |\n|  3 | 69.297  | 24.6529  | 44.3112 | 44.6441 | 101.868  | 11.2115  | 0.369345 | 23.5603 | 12.7074 | 11.4245  | -30.4702 | 18.8329 |           1 |\n|  4 | 49.7129 |  9.65207 | 28.3174 | 40.0608 | 108.169  |  7.9185  | 0.54336  | 35.494  | 15.9546 |  8.87237 | -16.3784 | 24.9171 |           1 |

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
