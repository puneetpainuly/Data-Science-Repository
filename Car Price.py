#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


inp0=pd.read_csv(r"C:\Users\puneet.painuly\Downloads\carprice.csv")


# In[38]:


inp0.head()


# In[40]:


inp0.shape


# In[41]:


inp0.info()


# In[42]:


inp0.sum()


# In[43]:


inp0.isna()


# In[44]:


inp0.describe()


# In[46]:


inp0.isnull().sum()


# In[47]:


inp0["CarName"].unique()


# In[48]:


inp0['CarName'] = inp0['CarName'].str.split(' ',expand=True)[0]


# In[49]:


inp0['CarName'].unique()


# In[50]:


inp0['CarName'] = inp0['CarName'].replace({'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 'vokswagen': 'volkswagen', 'vw': 'volkswagen'})


# In[51]:


inp0['CarName'].unique()


# In[52]:


inp0["CarName"].nunique()


# In[53]:


inp0["CarName"].value_counts()


# In[54]:


inp0.info()


# In[55]:


inp0.describe().round(2)


# In[56]:


inp0["price"].describe().round(2)


# In[60]:


plt.figure(figsize=(15,15))
ax=sns.countplot(x=inp0["CarName"]);
ax.bar_label(ax.containers[0]);
plt.xticks(rotation=90);


# In[62]:


plt.figure(figsize=(6,6))
sns.distplot(inp0["price"], hist=True);


# In[63]:


ax=sns.countplot(x=inp0["fueltype"]);
ax.bar_label(ax.containers[0]);


# In[64]:


ax=sns.countplot(x=inp0["doornumber"]);
ax.bar_label(ax.containers[0]);


# In[65]:


plt.figure(figsize=(15,15))
ax=sns.countplot(x=inp0["horsepower"]);
ax.bar_label(ax.containers[0]);
plt.xticks(rotation=90);


# In[69]:


plt.figure(figsize=(15,15))
sns.heatmap(inp0.corr(),annot=True);


# In[70]:


sns.pairplot(inp0)


# In[71]:


inp0.head()


# In[72]:


inp0.drop(columns = ['car_ID'], axis = 1, inplace = True)


# In[73]:


inp0['doornumber'] = inp0['doornumber'].map({'two': 2, 'four': 4})
inp0['cylindernumber'] = inp0['cylindernumber'].map({'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12})


# In[74]:


X = inp0.drop(columns = 'price', axis = 1)
y = inp0['price']


# In[75]:


X = pd.get_dummies(X, drop_first = True)
X.head()


# In[80]:


X.shape


# In[81]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif['Features'] = X.columns

vif


# In[83]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[84]:


print("Traing Data Shape of x and y respectively:  ", X_train.shape, y_train.shape)
print("Testing Data Shape of x and y respectively:  ", X_test.shape, y_test.shape)


# In[85]:


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[86]:


y_pred = lr_model.predict(X_test)


# In[87]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mse = mean_squared_error(y_pred, y_test)
r2_score = r2_score(y_pred, y_test)


# In[88]:


lr_model.score(X_test, y_test)


# In[89]:


mse


# In[90]:


rmse = np.sqrt(mse)
rmse


# In[91]:


r2_score


# In[ ]:





# In[ ]:




