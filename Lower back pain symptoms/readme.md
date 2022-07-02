
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
- **Class_att** contains two unique categories; <code>abnormal</code> : 1 <code>normal</code> 0

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

```
|    | Col1        | Col2        | Col3        | Col4        | Col5        | Col6         | Col7        | Col8    | Col9    | Col10    | Col11      | Col12   | Class_att   |
|:---|:------------|:------------|:------------|:------------|:------------|:-------------|:------------|:--------|:--------|:---------|:-----------|:--------|:------------|
| 0  | 63.0278175  | 22.55258597 | 39.60911701 | 40.47523153 | 98.67291675 | -0.254399986 | 0.744503464 | 12.5661 | 14.5386 | 15.30468 | -28.658501 | 43.5123 | 1.0         |
| 1  | 39.05695098 | 10.06099147 | 25.01537822 | 28.99595951 | 114.4054254 | 4.564258645  | 0.415185678 | 12.8874 | 17.5323 | 16.78486 | -25.530607 | 16.1102 | 1.0         |
| 2  | 68.83202098 | 22.21848205 | 50.09219357 | 46.61353893 | 105.9851355 | -3.530317314 | 0.474889164 | 26.8343 | 17.4861 | 16.65897 | -29.031888 | 19.2221 | 1.0         |
| 3  | 69.29700807 | 24.65287791 | 44.31123813 | 44.64413017 | 101.8684951 | 11.21152344  | 0.369345264 | 23.5603 | 12.7074 | 11.42447 | -30.470246 | 18.8329 | 1.0         |
| 4  | 49.71285934 | 9.652074879 | 28.317406   | 40.06078446 | 108.1687249 | 7.918500615  | 0.543360472 | 35.494  | 15.9546 | 8.87237  | -16.378376 | 24.9171 | 1.0         |
```

### Data Preparation
- Quite straightforward **0.8**/**0.2** train/test split of the dataset with <code>suffle</code>
- Standardise dataset; rescale the distribution of values so that the <code>mean</code> is 0 and the <code>standard deviation</code> is 1

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

```

X_train: (248, 12)
X_val: (62, 12)

```

- Create <code>tensors</code> of type <code>float</code> of all split data using <code>train_test_split</code>
- Create datasets containing <code>tensors</code>
- Create <code>DataLoader</code> with a <code>batch_size</code> 16 & <code>shuffle</code> option

```Python

X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)
y_train_tensor = torch.FloatTensor(y_train)
y_val_tensor = torch.FloatTensor(y_val)

# Builds dataset containing ALL data points
train_dataset = TensorDataset(X_train_tensor,
                              y_train_tensor)
val_dataset = TensorDataset(X_val_tensor,
                            y_val_tensor)

# Builds a loader of each set
batch_size = 16
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, 
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size)

```
