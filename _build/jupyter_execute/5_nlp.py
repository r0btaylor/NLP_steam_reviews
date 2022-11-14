#!/usr/bin/env python
# coding: utf-8

# (references:NLP)=
# # Natural Language Processing
# 
# The following natural language processing pipeline is performed to clean and process text ready for text-based exloratory analysis.
# 
# The NLP pipeline uses the `spaCy` package for Python {cite}`spacy_2020`.
# 
# <p align="center"> <img src="https://i.imgur.com/7zpgIyf.png"> </p>

# ## Load Data

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv('data/processed_review_data.csv',parse_dates=['date'])

# Train-test split. 20% test. Stratify on y label
X = df.drop(columns = ['classification'])
y = df[['classification']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = y)

# re-compile training set for eda
df = pd.concat([X_train, y_train], axis = 1).reset_index()
df.drop(columns = 'index',inplace=True)


# ## Text Clean-up
# 
# Here text is cleaned and prepared for the proceeding NLP pipeline.  
# At each step, a function is created and mapped to the `review_text` variable.

# `````{admonition} Note
# :class: tip
# All functions are also stored in a seperate file .py file to be called again later in model testing.
# `````

# ### Lower Case
# All capitalisation is removed from the text

# In[2]:


#define lower case function
def lower_case(text):
    text = text.lower()
    return text

# map to all review data
df = df.copy()
df['review_text'] = df['review_text'].astype(str).map(lower_case)

from pandas import option_context
with option_context('display.max_colwidth', 200):
    display(df[["review_text"]].head())


# ### Expand Contractions
# 
# A dictionary of expanded contractions is first created. This dictionary is used to replace all contractions with their expanded form.
# 
# (code modified from [here](https://gist.github.com/nealrs/96342d8231b75cf4bb82))

# In[3]:


import re

# define contractions dictionary
cList = {
    # A.
    "ain't": "am not","aren't": "are not",
    # C.
    "can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not",
    "couldnt": "could not","couldn't've": "could not have",
    # D.
    "didn't": "did not","doesn't": "does not","don't": "do not",
    # H.
    "hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would",
    "he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is","how'd": "how did",
    "how'd'y": "how do you","how'll": "how will","how's": "how is",
    # I.
    "i'd": "i would","i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have",
    "isn't": "is not","it'd": "it had","it'd've": "it would have","it'll": "it will","itll": "it will",
    "it'll've": "it will have","it's": "it is",
    # L.
    "let's": "let us",
    # M.
    "ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have",
    "must've": "must have","mustn't": "must not","mustn't've": "must not have",
    # N.
    "needn't": "need not","needn't've": "need not have",
    # O.
    "o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have",
    # S.
    "shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she would",
    "she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have",
    "shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is",
    # T.
    "that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there had",
    "there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have",
    "they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have",
    # V.
    "vr" : "virtual reality",
    # W.
    "wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
    "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have",
    "what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have",
    "where'd": "where did","where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have",
    "who's": "who is","who've": "who have","why's": "why is","why've": "why have","will've": "will have","won't": "will not",
    "won't've": "will not have","would've": "would have","wouldn't": "would not","wouldn't've": "would not have",
    # Y.
    "y'all": "you all","y'alls": "you alls","y'all'd": "you all would","y'all'd've": "you all would have",
    "y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have",
    "you'll": "you you will","you'll've": "you you will have","you're": "you are","you've": "you have"
}
c_re = re.compile('(%s)' % '|'.join(cList.keys()))

# define expansion function
def expandContractions(text, cList=cList):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

# map to all review data
df['review_text'] = df['review_text'].map(expandContractions)

with option_context('display.max_colwidth', 200):
    display(df[["review_text"]].head())


# ### Special Character Removal
# 
# All non alphabetical characters are removed from the review text.
# 
# (code modified from [here](https://www.techiedelight.com/remove-non-alphanumeric-characters-string-python/))

# In[4]:


# define non alpha removal function
def alpha_num(text):
    text = re.sub('[^a-zA-Z]',' ',text) #remove non-alpha
    text = re.sub(' +',' ', text) #remove extra whitespace
    return text

# map to all review data
df['review_text'] = df['review_text'].map(alpha_num)

with option_context('display.max_colwidth', 200):
    display(df[["review_text"]].head())


# ### Consecutive Removal
# 
# Identify and remove consecutive word duplicates and replace excessive duplicate characters (>2).

# In[5]:


# define duplicate removal function
from itertools import groupby

def consec_dup(text):
    text = " ".join([x[0] for x in groupby(text.split(" "))]) # remove repeat consecutive words
    text = re.sub(r'(.)\1+', r'\1\1',text) # replace >2 consecutive duplicate letters
    return text

# map to all review data
df['review_text'] = df['review_text'].map(consec_dup)

with option_context('display.max_colwidth', 200):
    display(df[["review_text"]].head())


# ### NLP
# 
# Here a natural language processing pipeline will be used.
# 
# Several actions will be performed:
# 
# **Stop Word Removal**
# A list of common words is supplied from the `nltk` module for Python {cite}`nltk_2020`. 
# 
# The frequency of these words is likely to be so high as to provide little analytical value and so they will be removed from the sample.
# 
# **Lemmatisation**
# Lemmatisation replaces all words with their base root form which will improve the proceeding text analysis.
# 
# For example: after lemmatisation, 'leaves' or 'leafs' would both be replaced with 'leaf'.
# 
# Lemmatisation is provided by  the `spaCy` module {cite}`spacy_2020`.
# 

# In[6]:


import nltk
from nltk.corpus import stopwords
from collections import Counter

# import stopwords and extend list
stpwrds = nltk.corpus.stopwords.words('english')
newStpWrds = ["game","play"]
stpwrds.extend(newStpWrds)

# create dictionary to increase processing speed
stpdict = Counter(stpwrds)

import spacy
nlp = spacy.load("en_core_web_sm")

def lemma(text):
    doc = nlp(text)
    text = [token.lemma_ for token in doc if token.text not in stpdict]
    text = " ".join(text)
    return text

df['review_text'] = df['review_text'].map(lemma)

with option_context('display.max_colwidth', 200):
    display(df[["review_text"]].head())


# A final check is made to remove entries with no review text.
# 
# Processing obviously now makes the `review_length` feature inaccurate and so it is dropped.
# 
# This concludes the creation of the final training data.
# 

# In[7]:


# quantify missing
display(pd.DataFrame({'Missing':len(df.loc[df['review_text'].str.split().str.len()<1]),
        'Present':len(df.loc[df['review_text'].str.split().str.len()>0]),
       'Total':len(df)},index = ['Review Text']))

# drop entries with no review text
df = df.drop(df[df['review_text'].str.split().str.len()<1].index)

# drop length variable
df.drop('review_length', axis = 1, inplace=True)

# write clean training data to csv
df.to_csv('data/train_data_lemma.csv',index=False)

df

