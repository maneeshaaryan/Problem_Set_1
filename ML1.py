
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import seaborn as sb


# In[3]:


data = pd.read_csv("boardgamegeek_dataset.csv")

print(data.columns)


# In[ ]:


data.head()


# In[4]:


data.shape


# In[5]:


data = data.drop(data.columns[0], axis=1)  # df.columns is zero-based pd.Index 


# In[6]:



import matplotlib.pyplot as plt


plt.hist(data["Average_Rating"])


# In[7]:


from sklearn.cluster import KMeans


kmeans_model = KMeans(n_clusters=5, random_state=1)

good_columns = data._get_numeric_data()

kmeans_model.fit(good_columns)

labels = kmeans_model.labels_


# In[8]:


from sklearn.decomposition import PCA

# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
# Show the plot.
plt.show()


# In[9]:


data.corr()["Average_Rating"]


# In[10]:


columns = data.columns.tolist()


# In[34]:


X_data = data[['Geek_Rating']]
y_data = data[['Average_Rating']]


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test= train_test_split(X_data,y_data,test_size =0.25,random_state=42)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[36]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[37]:


predictions=model.predict(X_test)


# In[40]:


sb.distplot(y_test-predictions)

