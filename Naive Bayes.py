#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[10]:


heart_df = pd.read_csv("heart.csv")
heart_df.head()


# In[12]:


X = heart_df.drop("target", axis = 1)
y = heart_df["target"]


# In[14]:


X.head()


# In[16]:


y.head()


# In[18]:


X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.2,random_state = 42
)


# In[22]:


# Naive Bayes Theorem
gnb_model = GaussianNB()
gnb_model.fit(X_train,y_train)


# In[24]:


y_pred = gnb_model.predict(X_test)


# In[30]:


# Evaluation
print("recall score:",recall_score(y_test,y_pred))
print("accuracy score:",accuracy_score(y_test,y_pred))
print("precision score:",precision_score(y_test,y_pred))


# In[ ]:




