#!/usr/bin/env python
# coding: utf-8

# In[19]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten


# In[2]:


(X_train,y_train),(X_test,y_test) = keras.datasets.mnist.load_data()


# In[3]:


X_test.shape


# In[4]:


y_train


# In[5]:


import matplotlib.pyplot as plt
plt.imshow(X_train[2])


# In[6]:


X_train = X_train/255
X_test = X_test/255


# In[7]:


X_train[0]


# In[8]:


model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[9]:


model.summary()


# In[10]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[11]:


history = model.fit(X_train,y_train,epochs=25,validation_split=0.2)


# In[12]:


y_prob = model.predict(X_test)


# In[13]:


y_pred = y_prob.argmax(axis=1)


# In[14]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[15]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[16]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


# In[17]:


plt.imshow(X_test[1])


# In[18]:


model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1)


# In[ ]:




