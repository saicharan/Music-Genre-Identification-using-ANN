#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import pandas as pd
import numpy as np


# In[2]:


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[3]:


#Keras
import keras


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# ## Analysing the Data 

# In[5]:


d=pd.read_csv('prj_data.csv')


# In[6]:


d.head()


# In[7]:


# Dropping unneccesary columns
d=d.drop(['filename'],axis=1)


# In[8]:


d.dtypes


# In[28]:


d.isnull().sum()


# In[9]:


# Encoding the Labels

types_list = d.iloc[:, -1]
onehot_le = LabelEncoder()
p = onehot_le.fit_transform(types_list)


# In[10]:


p


# In[29]:


d.head()


# In[11]:


# Scaling the Feature columns

di_std = StandardScaler()


# In[12]:


r = di_std.fit_transform(np.array(d.iloc[:, :-1], dtype = float))


# In[13]:


# Dividing data into training and Testing set

X_train, X_test, y_train, y_test = train_test_split(r, p, test_size=0.25)


# In[14]:


X_train[10]


# # Classification with Keras
# 

# In[15]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))


# In[16]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[17]:


history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)


# In[18]:


test_loss, test_acc = model.evaluate(X_test,y_test)


# In[19]:


print('test_acc: ',test_acc)


# 
# #### Let's set apart 200 samples in our training data to use as a validation set:

# In[20]:


x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]


# ### Now let's train our network for 30 epochs:

# In[21]:


model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=30,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)


# In[22]:


results


# ### Predictions on Test Data
# 

# In[23]:


predictions = model.predict(X_test)


# In[24]:


predictions[0].shape


# In[25]:


np.sum(predictions[0])


# In[26]:


np.argmax(predictions[0])


# In[ ]:




