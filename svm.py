#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


data=pd.read_csv('veri.csv')
data


# In[3]:


data = data[data['Rating'] != 3]
data['sentiment'] = data['Rating'].apply(lambda rating : +1 if rating > 3 else 0)


# In[4]:


df = data[['Review','sentiment']]
df.head()


# In[5]:


X = data.Review
y = data.sentiment


# In[6]:


vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)


# In[7]:


vect.fit(X_train)
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)


# In[8]:


SVM = LinearSVC()
SVM.fit(X_train_dtm, y_train)
y_pred = SVM.predict(X_test_dtm)
print('\nSupport Vector Machine')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')


# In[1]:


#R2 skoru ve Hassasiyet skoru
#print(f'R2 score: {r2_score(y_test,y_pred)}')
#print(f'Accuracy score: {accuracy_score(y_test,y_pred)}') 


# In[ ]:




