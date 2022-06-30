
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
- Create a tensor(s) from Numpy Arrays <code>X_train_tensor</code>, <code>y_train_tensor</code>, <code>X_val_tensor</code>, <code>y_val_tensor</code>
- Create a Dataset(s) (contains all data) <code>train_dataset</code>, <code>val_dataset</code>
- Create a DataLoader(s) (batch loading during training) <code>train_loader</code>, <code>val_loader</code>

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

#### Create a custom PyTorch class

```python

class pyRun(object):
    
    
    def __init__(self, model, loss_fn, optimizer):

        self.model = model       # Model
        self.loss_fn = loss_fn   # Loss Function
        self.optimizer = optimizer  # Optimiser
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)  # send to device 

        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        self.losses = []       # training loss data storage
        self.val_losses = []   # validation loss data storage
        self.total_epochs = 0  # total number of iteration loops

        # Creates the train_step function for our model

        self.train_step = self._make_train_step()  # create training step function
        self.val_step = self._make_val_step() # create validation step function

    # Used in minibatch
    def to(self, device):
        self.device = device
        self.model.to(self.device)

    # set dataloaders
    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    # set tensorboard
    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter('{}/{}_{}'.format(folder, name, suffix))
        
    ''' Private Classes '''

    # Training step
    # Makes use of global class attributes 
    
    def _make_train_step(self):
        
        # Builds function that performs a step in the train loop
        def perform_train_step(x, y):

            self.model.train()             # set model to train mode
            yhat = self.model(x)           # compute model prediction (forward pass)
            loss = self.loss_fn(yhat, y)   # compute the loss 
            loss.backward()                # compute gradients for both a,b parameters (backward pass)
            self.optimizer.step()          # update parameters using gradients and lr
            self.optimizer.zero_grad()     # reset gradients

            return loss.item()             # Returns the loss

        return perform_train_step # return the funct that will be called inside train loop
    
    def _make_val_step(self):
    
        # Performs a step in the validation loop
        def perform_val_step(x, y):
        
            self.model.eval() # set model to eval mode

            yhat = self.model(x)   # compute models predicted output (forward pass)
            loss = self.loss_fn(yhat, y)  # compute the loss function
            # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
            return loss.item()

        return perform_val_step
        
    # Mini Batch usable with both data loaders
        
    def _mini_batch(self, 
                    validation=False): # determines which step function to use
                    
        if validation:
            data_loader = self.val_loader
            step = self.val_step
        else:
            data_loader = self.train_loader
            step = self.train_step

        if data_loader is None:
            return None
            
        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train(self, n_epochs, seed=42):
        
        self.set_seed(seed)

        for epoch in range(n_epochs):

            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

        if self.writer:
            # Closes the writer
            self.writer.close()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
    
        checkpoint = torch.load(filename)  # load dict
        self.model.load_state_dict(checkpoint['model_state_dict'])  # restore state for model & optimiser
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train() # always use TRAIN for resuming training   

    def predict(self, x):
 
        self.model.eval()  # set to evaluation mode for predicitons
        x_tensor = torch.as_tensor(x).float() # numpy input -> float tensor
        y_hat_tensor = self.model(x_tensor.to(self.device)) # send input to device & predict
        self.model.train() # set back to train mode
        return y_hat_tensor.detach().cpu().numpy() # detaches it & brings to CPU & back to Numpy
    
    def plot_losses(self):
        
        fig = make_subplots(rows=1, cols=1,subplot_titles=None)
        itrace = go.Scatter(y=self.losses,marker_color='#283747',name='train')
        trace = go.Scatter(y=self.val_losses,marker_color='#C7C7C7',name='validation')

        fig.add_trace(itrace, row=1, col=1)
        fig.add_trace(trace, row=1, col=1)
        
        fig.update_layout(template='plotly_white',
                          title=f'Loss Function Convergence',
                          font=dict(family='sans-serif',size=12),
                          width=600)

        fig.update_traces(marker=dict(line=dict(width=0.5, color='white')),
                          opacity=0.75)
        fig.show()

    def add_graph(self):
        # Fetches a single mini-batch so we can use add_graph
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

```
