![](https://i.imgur.com/6zyuiVy.png)

### 1 | Chronic Lower Backpain

**[Chronic lower back pain (CLBP)](https://www.hopkinsmedicine.org/health/conditions-and-diseases/back-pain/7-ways-to-treat-chronic-back-pain-without-surgery)** is a major cause of disability worldwide. CLBP prevalence in adults has increased by more than 100% in the last decade and continues to rise in older populations (Allegri et al. 2016). Given, the complexity of lower back pain the severity of symptoms can differ from person to person. For this reason, **CLBP is often difficult to diagnose** requiring complex clinical decision-making, which can still result in misdiagnosis *[Allegri et al. 2016)]*

### 2 | Dataset Description

310 Observations, 13 Attributes (12 Numeric Predictors, 1 Binary Class Attribute - No Demographics)

Lower back pain can be caused by a variety of problems with any parts of the complex, interconnected network of spinal muscles, nerves, bones, discs or tendons in the lumbar spine. Typical sources of low back pain include:

- The large nerve roots in the low back that go to the legs may be irritated
- The smaller nerves that supply the low back may be irritated
- The large paired lower back muscles (erector spinae) may be strained
- The bones, ligaments or joints may be damaged
- An intervertebral disc may be degenerating
- An irritation or problem with any of these structures can cause lower back pain and/or pain that radiates or is referred to other parts of the body. > Many lower back problems also cause back muscle spasms, which don't sound like much but can cause severe pain and disability.

While lower back pain is extremely common, the symptoms and severity of lower back pain vary greatly. A simple lower back muscle strain might be excruciating enough to necessitate an emergency room visit, while a degenerating disc might cause only mild, intermittent discomfort.

### 3 | Purpose of this study

- Considering the clinical importance of lower back pain, I have chosen the lower back pain dataset which contains various measurements of physical spine data. 
- The goal of this analysis is to predict whether a patient will display abnormal (pain) or normal (no pain) given physical spine data.


### 4 | Import Dataset

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

- Check the <code>value_count</code> for <code>target</code> **Class_att**
- The class balance is slightly favoured towards **Abnormal**

```python
df['Class_att'].value_counts()
```

```

1    210
0    100
Name: Class_att, dtype: int64

```

### 5 | Data Preparation
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

- Data loader contains:

```python

for x,y in train_loader:
    print(x.shape,y)
    break
    
```

```

torch.Size([16, 12]) tensor([1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.])

```

### 6 | Define Model
- Define the neural network <code>classifier</code>
- The addition of <code>dropout</code> layers is quite a straightforward option in any neural network
  - From [source](https://www.kdnuggets.com/2019/12/5-techniques-prevent-overfitting-neural-networks.html)
  - > <code>Dropout</code> on the other hand, modify the network itself. It randomly drops neurons from the neural network during training in each iteration
- "The addition of <code>bach normalisation</code> from experimentation has been shown to improve the convergence properties (loss)"
  - From [source](https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd)
  - "**loss** of the network with <code>batch normalization</code> reduces much faster than the normal network because of the covariance shift"

```Python

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.layer_1 = nn.Linear(12, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

```

- Instantiate neural network class <code>Network</code>
- Set <code>optimiser</code> **Adam** with a <code>learning rate</code> of **1e-4**
- Set <code>loss function</code> **BCEWithLogitsLoss**

```Python

model = Network()
opt = optim.Adam(model.parameters(), lr=1e-4)
loss = nn.BCEWithLogitsLoss()

```

### 7 | Training Functions

- Define some helper functions for the main function <code>tain_val</code>

```Python

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

# get lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# loss function per batch
def loss_batch(loss_func, output, target, opt=None):
    
    loss = loss_func(output, target) # get loss
    
    if(opt is not None):
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss

# metrics

def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred)) # prediction 
    acc = accuracy_score(y_test,y_pred_tag.detach().numpy())
    acc = round(100*acc)
    return acc

def recall(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred)) # prediction 
    get_recall = recall_score(y_test,y_pred_tag.detach().numpy())
    get_recall = round(100*get_recall)
    return get_recall

def precision(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred)) # prediction 
    get_precision = precision_score(y_test,y_pred_tag.detach().numpy())
    get_precision = round(100*get_precision)
    return get_precision

def f1(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred)) # prediction 
    get_f1 = f1_score(y_test,y_pred_tag.detach().numpy())
    get_f1 = round(100*get_f1)
    return get_f1

# calculate metric & loss of entire dataset (epoch)
def loss_epoch(model,loss_func,dataset_dl,eval_funcs,opt=None):
    
    run_loss=0.0
    
    t_metric = {}; metric = {}
    for i in eval_funcs:
        t_metric[i] = 0.0
        
    # internal loop over dataset
    for xb, yb in dataset_dl:
        xb=xb.to(device)
        yb=yb.to(device)
        y_pred  = model(xb)
        
        loss = loss_batch(loss_func,y_pred, yb.unsqueeze(1),opt=opt)
        
        for feval in eval_funcs:
            if(feval == 'accuracy'):
                t_metric[feval] += accuracy(y_pred, yb.unsqueeze(1))
            if(feval == 'f1'):
                t_metric[feval] += f1(y_pred,yb.unsqueeze(1))
            if(feval == 'recall'):
                t_metric[feval] += recall(y_pred,yb.unsqueeze(1))
        
        run_loss += loss.item()
    loss=run_loss/len(dataset_dl)  # average loss value
    
    for feval in eval_funcs:
        temp = t_metric[feval]/len(dataset_dl)
        metric[feval] = temp  # average metric value
        
    
    return loss, metric
```

- Define the training function; **train_val**, requires arguments <code>model</code> (defined neural network) & <code>params</code> (training parameters)

```python

def train_val(model, params,verbose=False):
    
    epochs=params["epochs"]
    loss_func=params["f_loss"]
    opt=params["optimiser"]
    train_dl=params["train"]
    val_dl=params["val"]
    lr_scheduler=params["lr_change"]
    weight_path=params["weight_path"]
    eval_funcs = params['eval_func'] 
    write_metric = params['write_metric']
    
    loss_history={"train": [],"val": []} 
    best_model_wts = copy.deepcopy(model.state_dict()) 
    best_loss=float('inf') 
    
    tr_dict_eval = {}; te_dict_eval = {}
    for evals in eval_funcs:
        tr_dict_eval[evals] = []
        te_dict_eval[evals] = []
    
    for epoch in range(epochs):
        
        current_lr=get_lr(opt)
        model.train()
        train_loss, train_metric = loss_epoch(model,loss_func,train_dl,eval_funcs,opt)
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model,loss_func,val_dl,eval_funcs)
        
        if(val_loss < best_loss):
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
                
        loss_history["train"].append(train_loss)
        loss_history["val"].append(val_loss)
        
        for evals in eval_funcs:
            tr_dict_eval[evals].append(train_metric[evals])
            te_dict_eval[evals].append(val_metric[evals])
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if(verbose):
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 

        if(verbose):
            print(f"epoch: {epoch+1+0:03} | train loss: {train_loss:.3f} | val loss: {val_loss:.3f} | train-{write_metric}: {train_metric[write_metric]:.3f} val-{write_metric}: {val_metric[write_metric]:.3f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, {'train':tr_dict_eval,'val':te_dict_eval}

```

### 8 | Train the neural network model

- Set <code>device</code> which will be used in training
- Set <code>ReduceLROnPlateau</code> option to adjust the **learning rate** on the run
- Define training <code>params</code> dictionary
- Trained models via **train_val** output <code>model</code>, <code>loss data</code> <code>metric data</code>
- <code>write_metric</code> we'll set to the **f-measure**, specifically <code>f1</code>

```Python

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set ReduceLROnPlateau
lr_adapt = ReduceLROnPlateau(opt,mode='min',factor=0.5,patience=20,verbose=0)

# Create Parameter Dictionary
params_train={
  "train": train_loader,                               # Set Training Dataloader
    "val": val_loader,                                 # Set Validation Dataloader
 "epochs": 200,                                        # Number of Epochs
 "optimiser": opt,                                     # Set optimiser
 "lr_change": lr_adapt,                                # Set learning rate adapter
 "f_loss": loss,                                       # loss function 
 "weight_path": "weights.pt",                          # set weights path 
 "eval_func" : ['accuracy','f1','recall','precision'], # set metrics to be evaluated & saved
 "write_metric" : 'accuracy'                           # set metric to be printed on screen
}

# Train the model
nn_model,loss_hist,metric_hist=train_val(model,        # set model
                                         params_train, # set training parameters
                                         verbose=True) # option for verbose
epochs=params_train["epochs"] 

```

- Training process output with <code>verbose</code>

```

epoch: 176 | train loss: 0.185 | val loss: 0.330 | train-f1: 95.688 val-f1: 89.500
epoch: 177 | train loss: 0.186 | val loss: 0.332 | train-f1: 95.812 val-f1: 89.500
epoch: 178 | train loss: 0.176 | val loss: 0.328 | train-f1: 96.125 val-f1: 89.500
epoch: 179 | train loss: 0.180 | val loss: 0.330 | train-f1: 95.312 val-f1: 89.500
epoch: 180 | train loss: 0.173 | val loss: 0.319 | train-f1: 95.688 val-f1: 89.500
epoch: 181 | train loss: 0.216 | val loss: 0.336 | train-f1: 94.062 val-f1: 89.500
epoch: 182 | train loss: 0.209 | val loss: 0.314 | train-f1: 95.312 val-f1: 89.500
epoch: 183 | train loss: 0.170 | val loss: 0.315 | train-f1: 96.688 val-f1: 89.500
epoch: 184 | train loss: 0.200 | val loss: 0.331 | train-f1: 94.125 val-f1: 89.500
epoch: 185 | train loss: 0.191 | val loss: 0.324 | train-f1: 95.688 val-f1: 89.500
epoch: 186 | train loss: 0.191 | val loss: 0.332 | train-f1: 94.125 val-f1: 89.500
epoch: 187 | train loss: 0.226 | val loss: 0.327 | train-f1: 93.125 val-f1: 89.500
epoch: 188 | train loss: 0.193 | val loss: 0.317 | train-f1: 95.125 val-f1: 89.500
epoch: 189 | train loss: 0.198 | val loss: 0.322 | train-f1: 94.625 val-f1: 89.500
Saving best model weights!
epoch: 190 | train loss: 0.167 | val loss: 0.328 | train-f1: 97.312 val-f1: 89.500
epoch: 191 | train loss: 0.177 | val loss: 0.306 | train-f1: 96.688 val-f1: 90.750
epoch: 192 | train loss: 0.171 | val loss: 0.318 | train-f1: 96.312 val-f1: 89.500
epoch: 193 | train loss: 0.252 | val loss: 0.334 | train-f1: 93.312 val-f1: 89.500
epoch: 194 | train loss: 0.175 | val loss: 0.314 | train-f1: 96.688 val-f1: 89.500
epoch: 195 | train loss: 0.205 | val loss: 0.317 | train-f1: 96.250 val-f1: 89.500
epoch: 196 | train loss: 0.217 | val loss: 0.328 | train-f1: 93.688 val-f1: 89.500
epoch: 197 | train loss: 0.205 | val loss: 0.325 | train-f1: 93.312 val-f1: 89.500
epoch: 198 | train loss: 0.207 | val loss: 0.316 | train-f1: 94.000 val-f1: 89.500
epoch: 199 | train loss: 0.196 | val loss: 0.318 | train-f1: 95.062 val-f1: 89.500
epoch: 200 | train loss: 0.178 | val loss: 0.313 | train-f1: 95.625 val-f1: 89.500
Total Time: 18.311

```

### 9 | Visualise Training Results

<div style="color:white;display:fill;border-radius:8px;
            background-color:#03112A;font-size:150%;
            letter-spacing:1.0px">
    <p style="padding: 8px;color:white;"><b><b><span style='color:#94D4F6'>1.2 |</span></b> Visualise Results</b></p>
</div>

- Create a function that will plot the <code>loss</code> & <code>metric</code> stored in **loss_hist**,**metric_hist**
- During training, <code>precision</code>, <code>recall</code>, <code>f1</code>, <code>accuracy</code> are all stored, let's only view the harmonic mean, since the two classes are slightly <code>imbalanced</code>

```python

def plot_res(metric_hist,name):
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['lost_hist',f'metric_{name}'])

    # Training Data 

    fig.add_trace(go.Scatter(x=[*range(1,epochs+1)],
                             y=loss_hist["train"],
                              line=dict(color="#94D4F6",width=2),
                             name='train-loss'),row=1, col=1)
    fig.add_trace(go.Scatter(x=[*range(1,epochs+1)],
                             y=metric_hist["train"][name],
                             line=dict(color="#94D4F6",width=2),
                             name=f'train-{name}'),row=1, col=2)

    # Validation Data

    fig.add_trace(go.Scatter(x=[*range(1,epochs+1)],
                             y=loss_hist["val"],
                             line=dict(color="#454545",width=2),
                             name='val-loss'),row=1, col=1)
    fig.add_trace(go.Scatter(x=[*range(1,epochs+1)],
                             y=metric_hist["val"][name],
                             line=dict(color="#454545",width=2),
                             name=f'val-{name}'),row=1, col=2)

    fig.update_layout(template='plotly_white',
                      title='Train / Validation Data Splitting',
                      font=dict(family='sans-serif',size=12),
                      width=1200)

    fig.update_traces({'marker_line_width':3, 
                       'marker_line_color':"black",
                       'marker_size':8,
                       'opacity':1.0,
                       'marker':{'showscale':True,'reversescale':True, 'cmid':0, 'size':10},
                      })

    fig.update_coloraxes(colorscale="tealgrn")
    fig.update_layout(coloraxis_showscale=False)
    fig.show()

```

```python

plot_res(metric_hist,'f1')

```

![](https://i.imgur.com/xKg03zD.png)

### 10 | Inference 

- Let's check how well the data generalises on some test data, given the small number of rows in the data, let's just use part of the <code>validation</code> data, which works too, as we didn't train on the dataset
- We'll use a <code>threshold</code> of **0.5** & construct a <code>confusion matrix</code>

```python

X_test, y_test = val_dataset[:10]

model.eval() 
logits = nn_model(X_test.to(device)) 
probs = torch.sigmoid(logits)
probs # get probabilities 

threshold = .5
confusion_matrix(some_y[:10], (probs.cpu() >= threshold))

```

```
array([[4, 0],
       [0, 6]])
```

### 11 | Conclusion
- From background reading, **"CLBP is often difficult to diagnose"** (by humans), suggested there is a need to explore whether <code>neural networks</code> can fill in the void & replace humans in this task
- In this brief study we aimed at creating a <code>classifier</code> that could distinguish between two types <code>normal</code> & <code>abnormal</code>
- Our neural network <code>classifier</code> built using the <code>PyTorch</code> module allowed us to create a classifier that will be able to distinguish between the two types
- Due to imbalanced, we decided to focus on the <code>f1</code> metric score 
- Highest <code>generalisation</code> f1 score:
> epoch: 191 | train loss: 0.177 | val loss: 0.306 | train-f1: 96.688 val-f1: 90.750
- Such performance is a good start, however there is a need to improve the <code>classifier</code> model, some possible options:
  - Improvement of the neural network model (Adjust parameters that generalise <code>nn.Dropout(p=0.1)</code>, <Code>nn.BatchNorm1d(64)</code>)
  - Gain more data & retrain the model, the size can likely be a factor for poor generalisation performance on new data

### 12 | References
- Allegri et al., 2016, Mechanisms of low back pain: A guide for diagnosis and therapy. F1000Research, 5, 1530. doi:10.12688/f1000research.8105.1
