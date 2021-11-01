#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[9]:


from sklearn.datasets import load_iris
iris = load_iris()
iris


# In[10]:


iris_data = pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_data.head()


# In[11]:


iris_data['Speices'] = iris.target


# In[12]:


x =iris_data.iloc[:,:-1]
x


# In[13]:


y = iris_data.iloc[:,-1]
y


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[15]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)


# In[16]:


model.fit(x_train,y_train)


# In[17]:


y_pred = model.predict(x_test)


# In[18]:


y_pred


# In[19]:


y_test


# In[24]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('Accuracy_score:',accuracy_score(y_test,y_pred)*100,'%')
print('Precision_score:',precision_score(y_test,y_pred, average='macro')*100,'%')
print('recall_score:',recall_score(y_test,y_pred, average='macro')*100,'%')
print('F1_score:',f1_score(y_test,y_pred, average='macro')*100,'%')


# In[25]:


confusionmatrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusionmatrix, annot = True)
plt.title('IRIS')
plt.show()


# In[ ]:





# In[ ]:




