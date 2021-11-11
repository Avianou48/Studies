
# coding: utf-8

# ## Quiz 1 - Assignment

# #### Analyze and build machine learning model for the given dataset
# 
# + Research Questions - 10 pts
#     + List your research questions (at least 3 questions)
#     + Data info: Description about data source and reference
# 
# + Data Acquisition - 10 pts
#     + Information on dataset, data type (csv or other format) and research questions
#     + Information on data collection techniques and details
# 
# + Data Wrangling - 20 pts
#     + Data Wrangling strategy for answering your research questions
#     + Data Wrangling Code
#     + Provide code and detailed analysis along with some statistical insights for your underlying data
# 
# + EDA (Exploratory Data Analysis) - 15 pts
#     + Plot graphs/ charts/figures and add your observation for each plot. Also provide some statistical insights/ interpretation for your graphs/ plots
# 
# + Machine Learning Model - 30 pts
#     + Build machine learning model
#     + Evaluate the model
#     + Select and optimize the model
#     + Business justification for the ML model
# 
# + Conclusion - 15 pts
#     + Provide logical conclusion for your data analysis work aligning with your research questions
#     + Final conclusion of your data analysis work
# 

#  ### Research question:
#  #### How likely students who spend more time while studying are suceptible to pass? 
#  #### Do students with low income are seceptible to spend more time in study?
# 

# ### Data Acquisition

# In[19]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[20]:


df = pd.read_csv('Student-Pass-Fail-Data.csv')
data.head()


# In[21]:


df.shape


# In[22]:


#show the missing data number
df.isnull().sum()


# ### Exploratory Data Analysis

# In[23]:


sns.set_style('whitegrid')
sns.countplot(x='Pass_Or_Fail', data=df)


# In[24]:


df.columns


# In[25]:


sns.set_style('whitegrid')
sns.countplot(x='Self_Study_Daily', hue='Pass_Or_Fail', data=df)


# Observation: People that study at least six hours daily are most likely to pass while people who study less than six hours have almost no chance to pass.

# In[29]:


plt.figure(figsize=(10,7))
plt.bar(x=df['Self_Study_Daily'], height=df['Tution_Monthly'], color='red')
plt.show()


# Observation: People those montly tuition is over fourty spend less time studying than those that monthly tuition is less. 

# In[10]:


sns.heatmap(df.corr(),annot=True)


# Observation: There  is a strong positive correlation between self study daily and pass_or_fail rate.
#              There is a negative correlation between the monthly tuition student pay and the sucess or fail rate.

# In[32]:


X = df[['Self_Study_Daily', 'Tution_Monthly']]
X.head()


# In[31]:


y=df[['Pass_Or_Fail']]
y.head()


# ### Building Machine Learning models

# #### -- Logistic Regresion

# In[42]:


from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()


# In[43]:


from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
X_train, X_test, y_train, y_test = train_test_split(X,y)
print(X_train.shape,X_test.shape, y_train.shape, y_test.shape)


# In[44]:


logReg.fit(X_train,y_train)


# In[45]:


y_pred = logReg.predict(X_test)


# In[46]:


y_test.values[0:10]


# In[47]:


y_pred[0:10]


# In[48]:


logReg.score(X_test,y_test)


# In[49]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[50]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[57]:


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='0.0f')
plt.ylabel('actual value')
plt.xlabel('predicted value')
plt.title(f'Logistic Regression-Confusion Matrix')


# In[53]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# #### -- KNN Model

# In[58]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()


# In[60]:


scalar.fit(df.drop('Pass_Or_Fail', axis=1))


# In[61]:


scaled_features = scalar.transform(df.drop('Pass_Or_Fail', axis=1))


# In[63]:


std_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])
std_df.head()


# In[65]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['Pass_Or_Fail'])
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)


# In[67]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[68]:


knn.fit(x_train,y_train)


# In[69]:


y_predict = knn.predict(x_test)
y_predict


# In[71]:


score = knn.score(x_test,y_test)
score


# In[77]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_predict)
cm


# In[78]:


sns.heatmap(cm,annot=True,fmt='0.0f',cmap='GnBu')
plt.ylabel('actual value')
plt.xlabel('predicted value')
plt.title(f'KNN-Confusion Matrix')


# In[79]:


print(classification_report(y_test,y_predict))


# ## K value and Acuracy

# In[80]:


from sklearn import metrics
score = []


# In[81]:


for i in range(1,50):
  knn= KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  y_pred = knn.predict(x_test)
  score.append(metrics.accuracy_score(y_test,y_predict))

print(score)


# In[82]:


plt.figure(figsize=(10,8))
plt.plot(range(1,50), score, color='red')
plt.xlabel('KNN values')
plt.ylabel('Accuracy Score')


# In[83]:


error_rate = []
for i in range(1,40):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  pred_i = knn.predict(x_test)
  error_rate.append(np.mean(pred_i !=y_test))


# In[84]:


plt.figure(figsize=(10,8))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize =10)
plt.xlabel('KNN values')
plt.ylabel('Eror Rate'), 


# #### Conclusion:
#    - According to the result from the above the three models have similar outcomes. Logistic regression has a precision of 100% with 98% of accuracy and Knn Model has a precision of 99% with the same accuracy rate, but Knn has a best recall than the logistic regression model.
#    - So, both models are good since they are able to estimate a very close prediction to the testing dataset. 
#    - KNN is the most efficient approach with best k value between 16 and 38.
#     
#     

# In[ ]:




