#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
data=pd.read_csv('veri.csv')


# In[21]:


data


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px


# In[23]:


fig = px.histogram(data, x="Rating")
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Skor')
fig.show()


# In[5]:


data = data[data['Rating'] != 3]
data['sentiment'] = data['Rating'].apply(lambda rating : +1 if rating > 3 else 0)


# In[6]:


data


# In[7]:


data['sentimentt'] = data['sentiment'].replace({0 : 'negative'})
data['sentimentt'] = data['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(data, x="sentimentt")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1)
fig.update_layout(title_text='Product Sentiment')
fig.show()


# In[8]:


data['Summary'] = 'default value'
data


# In[9]:


def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final
data['Review'] = data['Review'].apply(remove_punctuation)
data = data.dropna(subset=['Summary'])
data['Summary'] = data['Summary'].apply(remove_punctuation)


# In[10]:


df = data[['Review','sentiment']]
df.head()


# In[11]:


import numpy as np
# rastgele train test datasını ayır
index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]


# In[12]:


# count vectorizer:
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Review'])
test_matrix = vectorizer.transform(test['Review'])


# In[13]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[14]:


X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']


# In[15]:


lr.fit(X_train,y_train)


# In[16]:


predictions = lr.predict(X_test)


# In[17]:


# accuracy, precision, recall:
from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
confusion_matrix(predictions,y_test)


# In[18]:


print(classification_report(predictions,y_test))


# In[ ]:




