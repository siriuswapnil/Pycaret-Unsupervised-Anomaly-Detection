#!/usr/bin/env python
# coding: utf-8

# - converted all headers to relevant headers
# - dropped unnecessary columns
# - applied frequency encoding with help of pycaret
# - applied isolated forest algorithm for anomaly detection
# - future trial -> one hot encoding with PCA, then apply autoencoders/ some supervised learning method

# In[1]:


pip install pycaret


# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('orig_data.csv')


# In[4]:


df.head(10)


# In[5]:


df.rename(columns = {'console' : 'accessingID', '/agent1/console' : 'accessingAddress', 'nk' : 'accessingType', 'room1' : 'sourceLocation', '/agent1' : 'accessedServiceAddress', 'nk.1' : 'accessedServiceType', 'room1.1' : 'destinationLocation', '/agent1.1' : 'accessedNodeAddress', 'nk.2' : 'accessedNodeType', 'read' : 'operation', 'Unnamed: 10' : 'value', '1617872464330' : 'timestanp', 'none' : 'valueTimestamp' }, inplace = True)


# In[6]:


df.head()


# In[15]:


df = df.drop(['timestanp'], axis = 1)


# In[16]:


df.head(20)


# In[17]:


df.info()


# In[18]:


for col in df.columns[0:]:
    print(col, ': ', len(df[col].unique()), ' labels')


# In[20]:


df.shape


# In[21]:


data = df.sample(frac=0.95, random_state=786)


# In[22]:


data_unseen = df.drop(data.index)


# In[23]:


data.reset_index(drop=True, inplace=True)


# In[24]:


data_unseen.reset_index(drop=True, inplace=True)


# In[25]:


print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[26]:


list(df.columns)


# In[28]:


from pycaret.anomaly import *


# In[29]:


exp_ano101 = setup(data, normalize = True, session_id = 123, high_cardinality_features = ['accessingID',
 'accessingAddress',
 'sourceLocation',
 'accessedServiceAddress',
 'destinationLocation',
 'accessedNodeAddress',
 'operation',
 'value'], high_cardinality_method = 'frequency')


# In[30]:


data


# In[31]:


iforest = create_model('iforest')


# In[32]:


print(iforest)


# In[33]:


iforest_results = assign_model(iforest)
iforest_results.head()


# In[46]:


iforest_results.head(50)


# In[34]:


iforest_anomaly = iforest_results.loc[iforest_results['Anomaly'] == 1]


# In[35]:


iforest_anomaly.shape


# In[36]:


iforest_anomaly.head(25)


# In[50]:


plot_model(iforest)


# In[27]:


plot_model(iforest, plot = 'umap')


# In[40]:


cluster = create_model('cluster')


# In[42]:


print(cluster)


# In[45]:


plot_model(cluster, plot = 'umap')


# In[46]:


knn = create_model('knn')


# In[47]:


print(knn)


# In[49]:


plot_model(knn, plot = 'umap')


# In[ ]:




