#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, string
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import numpy as np 
import pandas as pd 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score
import os
from sklearn.naive_bayes import BernoulliNB,MultinomialNB


# In[2]:


data=pd.read_csv('veri.csv')
data


# In[3]:


dist = data['Rating'].value_counts()
fig = px.bar(data, x=dist.index, y=dist.values,labels={'x':'Ratings','y': 'Sayısı'})
fig.show()


# In[15]:


x = data["Review"].copy()
y = data["Rating"].copy()


# In[21]:


def clean_text(review):
    review = review.lower()
    review = re.sub('\[.*?\]', '', review)
    review = re.sub('[%s]' % re.escape(string.punctuation), '', review)
    review = re.sub('\w*\d\w*', '', review)
    review = re.sub('[‘’“”…]', '', review)
    review = re.sub('\n', '', review)
   # stop_words=set(stopwords.words('english'))
    return review


round1 = lambda x: clean_text(x)
df = x.apply(round1)


# In[23]:


tfidfconverter = TfidfVectorizer(max_features=400, min_df=0.05, max_df=0.9)
tfidf = tfidfconverter.fit_transform(df).toarray()


# In[24]:


X_train,X_test,y_train,y_test = train_test_split(tfidf,y,test_size=0.2,random_state=42)


# In[25]:


nb = MultinomialNB()
nb_bern = BernoulliNB()


# In[26]:


nb_model=nb.fit(X_train,y_train)
nb_bern_model=nb_bern.fit(X_train,y_train)


# In[27]:


y_pred=nb_model.predict(X_test)
y_pred2=nb_bern_model.predict(X_test)


# In[28]:


print("Accuracy Multinominal:",accuracy_score(y_test, y_pred))
print("Accuracy Bernoulli:",accuracy_score(y_test, y_pred2))


# In[ ]:




