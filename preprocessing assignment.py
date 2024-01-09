#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('titanic_dataset.csv')


# In[3]:


data


# In[4]:


data.columns


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[11]:


sns.heatmap(data=data.corr(),annot=True)


# In[15]:


data.set_index('PassengerId')


# In[17]:


data.fillna(method='ffill', inplace=True)


# In[19]:


data


# In[52]:


data.isnull().sum()


# In[53]:


data.describe()


# In[54]:


sns.barplot(x='Pclass', y= 'Embarked', data = data,palette='viridis')


# In[55]:


sns.barplot(x='Sex', y= 'Pclass', data = data,palette='viridis')


# In[60]:


data.head()


# In[62]:


q1, q3 = np.percentile(data['Age'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = data[(data['Age'] >= lower_bound) 
                & (data['Age'] <= upper_bound)]
 


# In[64]:


q1, q3 = np.percentile(data['Fare'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = data[(data['Fare'] >= lower_bound) 
                & (data['Fare'] <= upper_bound)]


# In[65]:


q1, q3 = np.percentile(data['Survived'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = data[(data['Survived'] >= lower_bound) 
                & (data['Survived'] <= upper_bound)]


# In[66]:


q1, q3 = np.percentile(data['SibSp'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = data[(data['SibSp'] >= lower_bound) 
                & (data['SibSp'] <= upper_bound)]


# In[67]:


q1, q3 = np.percentile(data['Parch'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = data[(data['Parch'] >= lower_bound) 
                & (data['Parch'] <= upper_bound)]


# In[68]:


q1, q3 = np.percentile(data['Pclass'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
clean_data = data[(data['Pclass'] >= lower_bound) 
                & (data['Pclass'] <= upper_bound)]


# In[70]:


corr = data.corr()
sns.heatmap(data=data.corr(),annot=True)


# In[115]:


data


# In[150]:


data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
data = data.drop(columns='Name')

data['Title'] = data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
data['Title'] = data['Title'].replace('Mlle', 'Miss')

data['Title'] = data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})


# In[154]:


data['Sex'] = data['Sex'].map({'male':0, 'female':1})
data['Embarked'] = data['Embarked'].map({'C':0, 'Q':1, 'S':2})


# In[155]:


X = data.drop(columns='Survived')
y = data.Survived


# In[156]:


from sklearn.preprocessing import MinMaxScaler


# In[157]:


scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
rescaledX[:5]


# In[158]:


from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
rescaledX[:5]


# In[ ]:




