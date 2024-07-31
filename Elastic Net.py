#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nibabel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# In[2]:


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


# In[20]:


dataset = MRIData('PTs_500_4k_blinded.csv')
(images, labels, labels_PT4000) = dataset.get_data()
labels_pool=np.column_stack((labels, labels_PT4000))
train_imgs, test_imgs, train_labels_pool, test_labels_pool = train_test_split(images, labels_pool, test_size=0.2, random_state=0)
train_labels = train_labels_pool[:,0]
test_labels = test_labels_pool[:,0]
train_labels_4k = train_labels_pool[:,1]
test_labels_4k = test_labels_pool[:,1]


# In[22]:


print(train_labels.shape)
print(train_imgs.shape)
print(test_labels.shape)
print(test_imgs.shape)
n_train=train_labels.shape[0]
n_test=test_labels.shape[0]
print(f'Training data: {n_train}')
print(f'Test data: {n_test}')


# In[23]:


from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
import joblib
import time


# In[24]:


##choose the best alpha using 5-fold cross validation
#elastic_net_cv = ElasticNetCV(l1_ratio=0.5, alphas=np.arange(0.02, 0.31, 0.02), cv=5)
#elastic_net_cv.fit(train_imgs.reshape((n_train, -1)), train_labels)
#best_alpha = elastic_net_cv.alpha_


# In[25]:


best_alpha = 0.16


# In[7]:


start_time = time.time()
elastic_net = ElasticNet(alpha=best_alpha, l1_ratio=0.5)
elastic_net.fit(train_imgs.reshape((n_train, -1)), train_labels)
y_pred = elastic_net.predict(test_imgs.reshape((n_test, -1)))
mse = mean_squared_error(test_labels, y_pred)
r=np.corrcoef(y_pred, test_labels)[0, 1]
y_pred_train = elastic_net.predict(train_imgs.reshape((n_train, -1)))
mse_train = mean_squared_error(train_labels, y_pred_train)
r_train=np.corrcoef(y_pred_train, train_labels)[0, 1]
end_time = time.time()
elapsed_time = end_time - start_time
print(f'ElasticNet time spent: {elapsed_time}')
print(f'ElasticNet the best alpha from cross validation: {best_alpha}')
print(f'ElasticNet Test set Mean Squared Error: {mse}')
print(f'ElasticNet Test set Pearson Correlation: {r}')
print(f'ElasticNet Training set Mean Squared Error: {mse_train}')
print(f'ElasticNet Training set Pearson Correlation: {r_train}')
joblib.dump(elastic_net, 'Model1_elastic_net.pkl')

##plot
plt.figure(figsize=(8, 6))
plt.scatter(test_labels, y_pred, color='blue', alpha=0.5)
coefficients = np.polyfit(test_labels, y_pred, 1)
polynomial = np.poly1d(coefficients)
plt.plot(test_labels, polynomial(test_labels), color='red', linestyle='--')
plt.text(0.1, 0.9, 'r = {:.2f}'.format(r), transform=plt.gca().transAxes, fontsize=12)
plt.title('Scatter Plot of Observed vs. Predicted hearing loss at 500')
plt.xlabel('Observed hearing loss')
plt.ylabel('Predicted hearing loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('scatter_plot_500_model1.pdf')
# In[26]:


##train elasticNet for 4k
start_time = time.time()
elastic_net = ElasticNet(alpha=best_alpha, l1_ratio=0.5)
elastic_net.fit(train_imgs.reshape((n_train, -1)), train_labels_4k)
y_pred = elastic_net.predict(test_imgs.reshape((n_test, -1)))
mse = mean_squared_error(test_labels_4k, y_pred)
r=np.corrcoef(y_pred, test_labels_4k)[0, 1]
y_pred_train = elastic_net.predict(train_imgs.reshape((n_train, -1)))
mse_train = mean_squared_error(train_labels_4k, y_pred_train)
r_train=np.corrcoef(y_pred_train, train_labels_4k)[0, 1]
end_time = time.time()
elapsed_time = end_time - start_time
print(f'ElasticNet time spent: {elapsed_time}')
print(f'ElasticNet the best alpha from cross validation: {best_alpha}')
print(f'ElasticNet Test set Mean Squared Error: {mse}')
print(f'ElasticNet Test set Pearson Correlation: {r}')
print(f'ElasticNet Training set Mean Squared Error: {mse_train}')
print(f'ElasticNet Training set Pearson Correlation: {r_train}')
joblib.dump(elastic_net, 'Model1_elastic_net_model_4k.pkl')

##plot
plt.figure(figsize=(8, 6))
plt.scatter(test_labels_4k, y_pred, color='blue', alpha=0.5)
coefficients = np.polyfit(test_labels_4k, y_pred, 1)
polynomial = np.poly1d(coefficients)
plt.plot(test_labels_4k, polynomial(test_labels_4k), color='red', linestyle='--')
plt.text(0.1, 0.9, 'r = {:.2f}'.format(r), transform=plt.gca().transAxes, fontsize=12)
plt.title('Scatter Plot of Observed vs. Predicted hearing loss at 4k')
plt.xlabel('Observed hearing loss')
plt.ylabel('Predicted hearing loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('scatter_plot_4k_model1.pdf')
