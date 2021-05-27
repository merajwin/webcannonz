#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[37]:


data = pd.read_csv("data.csv",',',error_bad_lines=False)


# In[38]:


data.head()


# In[39]:


data[data['password'].isnull()]


# In[40]:


data.dropna(inplace=True)


# In[41]:


from sklearn.utils import shuffle
data = shuffle(data)


# In[42]:


data.reset_index(drop=True,inplace=True)


# In[43]:


y = data['strength']


# In[44]:


y.head()


# In[45]:


X = data['password']


# In[46]:


X.head()


# In[47]:


sns.set_style('whitegrid')
sns.countplot(x='strength',data=data)


# In[48]:


def words_to_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters


# In[49]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=words_to_char)
X=vectorizer.fit_transform(X)


# In[50]:


X.shape


# In[51]:


X.todense()


# In[52]:


vectorizer.vocabulary_


# In[53]:


data.iloc[0][0]


# In[54]:


feature_names = vectorizer.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=X[0]
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)


# In[55]:


## Logistics Regression

from sklearn.linear_model import LogisticRegression


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  #splitting


# In[58]:


log_class=LogisticRegression(penalty='l2',multi_class='ovr')
log_class.fit(X_train,y_train)


# In[59]:


print(log_class.score(X_test,y_test))


# In[60]:


## Multinomial

clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='saga')
clf.fit(X_train, y_train) #training
print(clf.score(X_test, y_test))


# In[61]:


X_predict=np.array(["2DFSabc#d$$$$"])
X_predict=vectorizer.transform(X_predict)
y_pred=log_class.predict(X_predict)
print(y_pred)


# In[62]:


import pickle

pickle.dump(clf, open("Meraj_Ahmed_20MAI0076.pkl","wb"))


# In[63]:


loaded_model = pickle.load(open("Meraj_Ahmed_20MAI0076.pkl", "rb"))
loaded_model.predict(X_test)
loaded_model.score(X_test,y_test)


# In[64]:


import tkinter as tk

from tkinter import ttk

win = tk.Tk()

win.title('Gimme some text!')


# In[65]:


data.head()


# In[66]:


#Column 1 
Preg=ttk.Label(win,text="password")
Preg.grid(row=0,column=0,sticky=tk.W)
Preg_var=tk.StringVar()
Preg_entrybox=ttk.Entry(win,width=16,textvariable=Preg_var)
Preg_entrybox.grid(row=0,column=1)


# In[68]:


import pandas as pd
DF = pd.DataFrame()
def action():
    global DB
    import pandas as pd
    DF = pd.DataFrame(pd.series(['password']))
    PREG=Preg_var.get()
    DF.loc[0,'password']=password
print(DF.shape)
DB=DF


# In[ ]:




