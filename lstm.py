#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px


# In[3]:


data=pd.read_csv('veri.csv')
data['sentiment'] = data['Rating'].apply(lambda rating : +1 if rating >= 3 else 0)


# In[4]:


data


# In[5]:


data['sentimentt'] = data['sentiment'].replace({0 : 'negative'})
data['sentimentt'] = data['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(data, x="sentimentt")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1)
fig.update_layout(title_text='Product Sentiment')
fig.show()


# In[6]:


data = data[['Review','sentimentt']]


# In[7]:


data['Review'] = data['Review'].apply(lambda x: x.lower())
data['Review'] = data['Review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentimentt'] == 'positive'].size)
print(data[ data['sentimentt'] == 'negative'].size)

#for idx,row in data.iterrows():
#    row[0] = row[0].replace('',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['Review'].values)
X = tokenizer.texts_to_sequences(data['Review'].values)
X = pad_sequences(X)


# In[8]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[9]:


Y = pd.get_dummies(data['sentimentt']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape) 


# In[13]:


batch_size = 32
history = model.fit(X_train, Y_train, epochs = 1, batch_size=batch_size, verbose = 2)
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train_n,Y_oneh,epochs=10,batch_size=64)


# In[14]:


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# In[15]:


import matplotlib.pyplot as plt
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,label='Eğitim Kayıp')
plt.plot(epochs,val_loss,label='Test Kayıp')
plt.title('Eğitim ve Test Kayıp Değerleri')
plt.xlabel('Deneme')
plt.ylabel('Kayıp')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Doğruluk Grafiği')
plt.ylabel('Doğruluk')
plt.xlabel('Deneme')
plt.legend(['train','test'], loc='upper right')
plt.show
acc=model.evaluate(x_train,y_train)
print("Loss:",acc[0],"Accuracy:",acc[1])
#pred= model.predict(x_test)
#pred_y=pred.argmax(axis=-1)
#cm=confusion_matrix(y_test,pred_y)
#sn.heatmap(cm,annot=True)


# In[ ]:




