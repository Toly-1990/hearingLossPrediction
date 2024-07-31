#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nibabel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# In[3]:


class MRIData():
    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(self.file)
        self.df.ID = self.df.ID.str.replace('_', '')
    def get_img(self, filename):
        img = nibabel.load(f'./dataset/{filename}').get_fdata()
        img = np.array(img)
        return img
    def get_data(self):
        n = self.df.shape[0]
        images = np.array([self.get_img(self.df['ID'][i]+'_T1.nii') for i in range(n)], dtype=object)
        labels =self.df['PT500']
        labels_PT4000 =self.df['PT4000']
        return images, labels, labels_PT4000


# In[4]:


dataset = MRIData('PTs_500_4k_blinded.csv')
(images, labels, labels_PT4000) = dataset.get_data()
labels_pool=np.column_stack((labels, labels_PT4000))
train_imgs, test_imgs, train_labels_pool, test_labels_pool = train_test_split(images, labels_pool, test_size=0.2, random_state=0)
train_labels = train_labels_pool[:,0]
test_labels = test_labels_pool[:,0]
train_labels_4k = train_labels_pool[:,1]
test_labels_4k = test_labels_pool[:,1]
print(train_labels.shape)
print(train_imgs.shape)
print(test_labels.shape)
print(test_imgs.shape)
n_train=train_labels.shape[0]
n_test=test_labels.shape[0]
print(f'Training data: {n_train}')
print(f'Test data: {n_test}')


# In[6]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
import time


# In[ ]:


X_train_tensor = torch.FloatTensor(train_imgs.astype(np.float32)).unsqueeze(1)  # Add a channel dimension
y_train_tensor = torch.FloatTensor(train_labels)
X_test_tensor = torch.FloatTensor(test_imgs.astype(np.float32)).unsqueeze(1)  # Add a channel dimension
y_test_tensor = torch.FloatTensor(test_labels)
y_train_tensor_4k = torch.FloatTensor(train_labels_4k)
y_test_tensor_4k = torch.FloatTensor(test_labels_4k)


# In[7]:


class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(27989648, 64)  # Adjust the input size based on the new architecture
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


# In[8]:


model = CNNRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# In[ ]:


# Training the CNN model
start_time = time.time()
num_epochs = 40
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}')


# In[ ]:


# Evaluate the CNN model
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()
    y_pred_train_tensor = model(X_train_tensor)
    y_pred_train = y_pred_train_tensor.numpy()

end_time = time.time()
elapsed_time = end_time - start_time
print(f'CNN time spent: {elapsed_time}')


mse = mean_squared_error(test_labels, y_pred)
r=np.corrcoef(y_pred.flatten(), test_labels)[0, 1]
mse_train = mean_squared_error(train_labels, y_pred_train)
r_train=np.corrcoef(y_pred_train.flatten(), train_labels)[0, 1]
print(f'CNN Test set Mean Squared Error: {mse}')
print(f'CNN Test set Pearson Correlation: {r}')
print(f'CNN Training set Mean Squared Error: {mse_train}')
print(f'CNN Training set Pearson Correlation: {r_train}')
torch.save(model.state_dict(), 'Model3_neural_network.pth')

##plot
plt.figure(figsize=(8, 6))
plt.scatter(test_labels, y_pred, color='blue', alpha=0.5)
coefficients = np.polyfit(test_labels, y_pred.flatten(), 1)
polynomial = np.poly1d(coefficients)
plt.plot(test_labels, polynomial(test_labels), color='red', linestyle='--')
plt.text(0.1, 0.9, 'r = {:.2f}'.format(r), transform=plt.gca().transAxes, fontsize=12)
plt.title('Scatter Plot of Observed vs. Predicted hearing loss at 500')
plt.xlabel('Observed hearing loss')
plt.ylabel('Predicted hearing loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('scatter_plot_500_model3.pdf')

# # train model for 4K data

# In[ ]:


model = CNNRegression()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1)

# Convert data to DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor_4k)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# In[ ]:


# Training the CNN model
start_time = time.time()
num_epochs = 40
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}')


# In[ ]:


# Evaluate the CNN model
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()
    y_pred_train_tensor = model(X_train_tensor)
    y_pred_train = y_pred_train_tensor.numpy()

end_time = time.time()
elapsed_time = end_time - start_time
print(f'CNN time spent: {elapsed_time}')


mse = mean_squared_error(test_labels_4k, y_pred)
r=np.corrcoef(y_pred.flatten(), test_labels_4k)[0, 1]
mse_train = mean_squared_error(train_labels_4k, y_pred_train)
r_train=np.corrcoef(y_pred_train.flatten(), train_labels_4k)[0, 1]
print(f'CNN Test set Mean Squared Error: {mse}')
print(f'CNN Test set Pearson Correlation: {r}')
print(f'CNN Training set Mean Squared Error: {mse_train}')
print(f'CNN Training set Pearson Correlation: {r_train}')
torch.save(model.state_dict(), 'Model3_neural_network_4k.pth')

##plot
plt.figure(figsize=(8, 6))
plt.scatter(test_labels_4k, y_pred, color='blue', alpha=0.5)
coefficients = np.polyfit(test_labels_4k, y_pred.flatten(), 1)
polynomial = np.poly1d(coefficients)
plt.plot(test_labels_4k, polynomial(test_labels_4k), color='red', linestyle='--')
plt.text(0.1, 0.9, 'r = {:.2f}'.format(r), transform=plt.gca().transAxes, fontsize=12)
plt.title('Scatter Plot of Observed vs. Predicted hearing loss at 4k')
plt.xlabel('Observed hearing loss')
plt.ylabel('Predicted hearing loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('scatter_plot_4k_model3.pdf')