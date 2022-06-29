'''

1D MLP from Numpy array (torch.from_numpy())
Regression
Custom Class model
Custom Dataset 

'''

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = torch.from_numpy(np.linspace(1,100,num=100)[:,None]).float()
y = torch.from_numpy(np.dot(2,x)).float()

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self):
        self.sequences = x
        self.target = y
        
    def __getitem__(self,i):
        return self.sequences[i], self.target[i]
    
    def __len__(self):
        return len(self.sequences)

# Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 1)
        
    def forward(self,inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to('cpu')

# Generators
training_set = MyDataset()
loader = torch.utils.data.DataLoader(training_set,
                                     batch_size=10)
# loss function and optimiser
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.0001)

# Training Iterations
for epoch in range(12):
    for inputs,target in loader:
        
        optimiser.zero_grad()   # Clears old gradients from last step
        output = model(inputs)  # Computes our model's predicted output - forward pass
        loss = criterion(output,target) # compute the loss
        
        loss.backward() # Computes gradients for both "a" and "b" parameters
        optimiser.step() # Updates parameters using gradients and the learning rate
