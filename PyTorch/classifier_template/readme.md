
#### Binary Classification Example

- We'll be utilising a simple dataset, containing two classes
- Our problem will be a <code>binary</code> classification problem


#### Create Dataset

- <code>sklearn</code> make_moons creates a <code>two</code> class data structure (<code>samples</code> 400, <code>noise</code> 0.8)
- Training/Test split is utilised (<code>test_size</code> 0.2, <code>shuffle</code> True)
- <code>scaling</code> will be used before importing the data into the neural network

```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400,
                  noise=0.8,
                  random_state=11)

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.2,
                                                  random_state=13,
                                                  shuffle=True)

sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_val = sc.transform(X_val)

```

#### Visualise Dataset

- Having split the data into a <code>training</code> and <code>validation</code> datasets, let's visualise it

```python

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2,subplot_titles=['Train','Validation'])
itrace = px.scatter(X_train,
                    color=y_train,
                   )['data'][0]
trace = px.scatter(X_val,
                   color=y_val)['data'][0]

fig.add_trace(itrace, row=1, col=1)
fig.add_trace(trace, row=1, col=2)

fig.update_layout(template='plotly_white',
                  title='Train / Validation Data Splitting',
                  font=dict(family='sans-serif',size=12),
                  width=1200)

fig.update_traces({'marker_line_width':1.5, 
                   'marker_line_color':"black",
                   'marker_size':8,
                   'opacity':1.0,
#                    'marker':{'showscale':True,'reversescale':True, 'cmid':0, 'size':10},
                  })

fig.update_coloraxes(colorscale="tealgrn")
fig.update_layout(coloraxis_showscale=False)
fig.show()

```

#### Create Data Loaders
- Create a tensor
- create a Dataset (contains all data)
- Create a DataLoader (batch loading)

```python

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Builds tensors from numpy arrays
x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train[:,None]).float()
x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val[:,None]).float()

# Builds dataset containing ALL data points
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

# Builds a loader of each set
batch_size = 10
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, 
                          shuffle=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size)
                        
```
