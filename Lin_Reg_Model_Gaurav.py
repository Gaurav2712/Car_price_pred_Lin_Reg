#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
A chinese car company wants to enter US market to counter European and US market 
they have contracted a car consulting company who have provided a set of data of cars 
here we will be predicting the price of cars while cosidering the factors on which the price of a car depends upon looking at
the previous data
'''


# In[ ]:


#  Linear regression model is used here to predict the price of the car.
#  It helps in finding a best fit which minimizes the difference between Actual Dependent var and predicted  variable


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#importing sklearn libraries 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#importing statsmodel api module
import statsmodels.api as sm


# In[4]:


# STEP 1: importing dataset
df=pd.read_csv("C:\\Users\\hp\\Downloads\\Cars_Retail_Price.csv")
df


# In[5]:


#Shape of data
df.shape


# In[6]:


analysis=df.describe()
pd.DataFrame(analysis)
analysis


# In[7]:


#Checking All variables
df.columns


# In[8]:


#Checking information regarding all variables
df.info()


# In[10]:


# exploring categorical variables
for tables in ['Cylinder','Doors','Cruise','Sound','Leather','Make','Model','Trim','Type']:
    df[tables]=df[tables].astype('category')
    
    
print(df)


# In[11]:


########## Now analysisng the categorical  columns
Cat_vars= df.select_dtypes(include='category').columns.tolist()
Cat_vars=list(Cat_vars)
Cat_vars


# In[12]:


for i in Cat_vars:
    x= df[i].value_counts()
    print(x)


# In[13]:


#Checking the Null Values %age:

round(((df.isna().sum() / df.shape[0]) * 100),2)


# In[14]:


df.info()


# In[15]:


# Data preparation
# creating dummy variables(for all categorical variables):
df['Make'].value_counts()
dummies = pd.get_dummies(df['Make']).rename(columns=lambda x: 'Make_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
df


# In[16]:


df['Model'].value_counts()
dummies2 = pd.get_dummies(df['Model']).rename(columns=lambda x: 'Model_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies2], axis=1)
df


# In[17]:


df['Trim'].value_counts()
dummies3 = pd.get_dummies(df['Trim']).rename(columns=lambda x: 'Trim_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies3], axis=1)
df


# In[18]:


df['Type'].value_counts()
dummies4 = pd.get_dummies(df['Type']).rename(columns=lambda x: 'Type_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies4], axis=1)
df


# In[19]:


df['Cylinder'].value_counts()
dummies5 = pd.get_dummies(df['Cylinder']).rename(columns=lambda x: 'Cylinder_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies5], axis=1)
df


# In[20]:


df['Cruise'].value_counts()
dummies6 = pd.get_dummies(df['Cruise']).rename(columns=lambda x: 'Cruise_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies6], axis=1)
df


# In[21]:


df['Leather'].value_counts()
dummies8 = pd.get_dummies(df['Leather']).rename(columns=lambda x: 'Leather_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies8], axis=1)
df


# In[22]:


df['Sound'].value_counts()
dummies9 = pd.get_dummies(df['Sound']).rename(columns=lambda x: 'Sound_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies9], axis=1)
df


# In[23]:


# using near zero variance removing the variables
y= np.var(df, axis=0)
y=pd.DataFrame(y)
y.to_csv('vars1.csv')


# In[24]:


new_df=df[['Price','Mileage','Liter','Cylinder_8','Cylinder_6','Cylinder_4','Sound_1','Sound_0','Cruise_0','Cruise_1','Make_Chevrolet','Trim_Sedan 4D',
          'Make_SAAB','Make_Buick','Type_Wagon','Make_Pontiac','Make_Saturn','Type_Hatchback','Type_Convertible','Trim_Coupe 2D',
           'Trim_LS Sedan 4D','Trim_LS Coupe 2D','Trim_Aero Sedan 4D','Model_AVEO','Model_Cavalier','Model_Malibu','Model_AVEO',
           'Model_Malibu','Model_AVEO','Model_Cobalt','Model_Ion','Model_Impala','Leather_1','Leather_0'
          ]]
new_df


# In[25]:


# dropping correlated variables
cor= new_df.corr()
cor.to_csv('cor_lin_reg1.csv')


# In[26]:


final_df=new_df.drop(['Model_Ion','Cylinder_6'],axis=1)
final_df


# In[27]:


model_df=final_df.apply(pd.to_numeric)


# In[ ]:


####################################### we are done with feature selection ####################
# Now we will split the dependent and independent variables 


# In[29]:


x= model_df.iloc[:,1:32]
y=model_df.iloc[:,0]
print(x)
print(y)


# In[35]:


#Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3) 


# In[37]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm = lm.fit(x_train,y_train)


# In[38]:


#To predict the values of y on the test set we use lm.predict( )

y_pred = lm.predict(x_train)
y_pred = pd.DataFrame(y_pred)
y_train= pd.DataFrame(y_train)


# In[39]:


print(y_pred)
print(y_train)


# In[40]:


y_train.to_csv('y_train.csv') 
y_pred.to_csv('y_pred.csv') 


# In[42]:


# actual prediction 
y_pred = lm.predict(x_test)
y_pred = pd.DataFrame(y_pred)
y_test= pd.DataFrame(y_test)


# In[43]:


# converting the file to csv format to analyse better
y_test.to_csv('y_test_t.csv') 
y_pred.to_csv('y_pred_t.csv') 
# here i analysed that there is an error of 10% between the testing and the perdictied price of cars and 
# hence model is good to predict the car prices so we can deploy  


# In[44]:


# how good our independent var are in predicting pri
r2_score(y_test,y_pred)
# this r2 score is satisfactory for our prediction


# In[45]:


#  pred on new data/ scoring /deployment:
z_test=x_test.sample(n=3)
new_car_prices=lm.predict(z_test)
new_car_prices


# In[46]:


z_test


# In[47]:


# test and prdicted value taking a linear relationship
plt.scatter(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




