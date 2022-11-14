#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#import numpy as np
from datetime import datetime
from textblob import TextBlob
import spacy

spacy.require_cpu()
nlp = spacy.load("en_core_web_sm")

# load data
df = pd.read_csv('data/processed_review_data.csv',parse_dates=['date'])

# Restrict to review >=10 words
df = df[df['review_length']>=10]


# In[2]:


# clean review text
from functions import lower_case,expandContractions,alpha_num,consec_dup,lemma
import re
def clean(text):
    text = re.sub(r'[?!:]', '.', text) # all sentence ends with '.'
    text = re.sub('\d*\.\d+','', text) # remove all flots
    text = re.sub("[^a-zA-Z0-9. ]", '', text.lower()) # remove all not listed chars and make lowercase
    text = re.sub('\.\.+', '. ',text) #remove repeat fullstops
    text = re.sub(' +',' ', text) # remove extra whitespace
    text = TextBlob(text)
    text = text.correct() # Correct spellings
    return text

for func in [expandContractions,clean,consec_dup,lemma]:
    df.review_text = df.review_text.map(func)


# In[3]:


# split text into sentences and flatten
sentences = [x.split('.') for x in df.review_text]
sentences = [item for sublist in sentences for item in sublist]


# In[4]:


# Extract aspects and descriptors
aspects = []
for sentence in sentences:
  doc = nlp(sentence)
  descriptive_term = ''
  target = ''
  for token in doc:
    if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
      target = token.text
    if token.pos_ == 'ADJ':
      prepend = ''
      for child in token.children:
        if child.pos_ != 'ADV':
          continue
        prepend += child.text + ' '
      descriptive_term = prepend + token.text
  aspects.append({'aspect': target,
    'description': descriptive_term})

# remove entries with blank aspect or descriptor
aspects = [x for x in aspects if x['aspect']!='' and x['description']!='']

# Add sentiment polarity scores
for aspect in aspects:
  aspect['sentiment'] = TextBlob(aspect['description']).sentiment.polarity

sent_df = pd.DataFrame(aspects)
sent_df


# In[5]:


sent_df.sort_values(by='sentiment',ascending = False).head(50)


# In[6]:


neutral = sent_df[sent_df['sentiment']==0]

neg = pd.read_csv("C:/Users/rob_t/OneDrive/Documents/Data Science/rMarkDown/SA_steam_reviews/data/negList.csv")
neg = list(neg['Negative'])
neg = list(neutral.loc[neutral['description'].isin(neg)].description+' '+neutral.loc[neutral['description'].isin(neg)].aspect)

pos = pd.read_csv("C:/Users/rob_t/OneDrive/Documents/Data Science/rMarkDown/SA_steam_reviews/data/posList.csv")
pos = list(pos['Positive'])
pos = list(neutral.loc[neutral['description'].isin(pos)].description+' '+neutral.loc[neutral['description'].isin(pos)].aspect)

