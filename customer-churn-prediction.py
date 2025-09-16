#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import os

# List available files in /kaggle/input
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the CSV (adjust filename to what you see in the printout above)
df = pd.read_csv('C:/Users/shanm/python DSA/Churn_Modelling.csv')

# Show the first few rows
df.head()


# In[4]:


import pandas as pd

df = pd.read_csv('C:/Users/shanm/python DSA/Churn_Modelling.csv')
df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True)
print(df.head())


# In[5]:


df.head()


# In[6]:


df['Geography'].value_counts()


# In[7]:


df['Gender'].value_counts()


# In[8]:


df = pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)


# In[9]:


df.head()


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[11]:


# Define features (X) and target (y)
X = df.drop('Exited', axis=1)   # Features
y = df['Exited']                # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Import and apply StandardScaler
scaler = StandardScaler()
X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)

print("X_train_trf shape:", X_train_trf.shape)
print("X_test_trf shape:", X_test_trf.shape)


# In[12]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense


# In[13]:


model = Sequential()

model.add(Dense(11,activation='sigmoid',input_dim=11))
model.add(Dense(11,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))


# In[14]:


model.summary()


# In[15]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[16]:


history = model.fit(X_train,y_train,batch_size=50,epochs=100,verbose=1,validation_split=0.2)


# In[17]:


y_pred = model.predict(X_test)


# In[18]:


y_pred


# In[19]:


y_pred = y_pred.argmax(axis=-1)


# In[20]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[21]:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()


# In[22]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# In[ ]:




