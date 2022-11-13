#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 
# ## Train & Test Splits
# 
# To prevent any data leakage during exploratory data analysis, the data was first split into training and testing sets.
# 
# The split was stratified according to the `classification` variable so that both training and testing sets retain representative distributions of the target label.
# 
# All exploratory analysis was performed on the training set only.

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


# ## Class Distribution
# 
# Graphing the frequency of each review category demonstrates some imbalance between the review classifications. 
# 
# Imbalance present is not extreme (60:40) and should not pose any particular challenge for model training. However, resampling will be investigated to ascertain whether this would impact results.  

# In[2]:


# data visualisation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

# plot positive vs negative reviews
sns.set_style('whitegrid')
ax = sns.countplot(x="classification", 
              data=df, 
              palette = ['#88D8B0', '#FF6F69'])

# add annotations
for c in ax.containers:
    labels = [f"{len(df[df['classification']=='Positive'])/(len(df[df['classification']=='Positive'])+len(df[df['classification']=='Negative']))*100:0.0f}%",
              f"{len(df[df['classification']=='Negative'])/(len(df[df['classification']=='Negative'])+len(df[df['classification']=='Positive']))*100:0.0f}%"]
    ax.bar_label(c, labels=labels, label_type='edge')

plt.title('Positive vs Negative Reviews',fontsize=14)
plt.xlabel('Classification',fontsize=12)
plt.ylabel('Number of reviews',fontsize=12)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.show();


# ## Reviews Over Time
# 
# The number of reviews has steadily declined since release. A spike in negative reviews on October 29th is notable.

# In[3]:


# plot quantity of pos and neg reviews over time
data = pd.DataFrame({'Count' : df.groupby( [ "date", "classification"] ).size()}).reset_index()
data['date'] = pd.to_datetime(data['date']).dt.strftime('%b %d')
sns.lineplot(data = data,
             x = 'date',
             y = 'Count',
             hue = 'classification',
            palette = ['#FF6F69','#88D8B0'])
plt.xticks(rotation=45)
plt.xlabel('')
plt.title('Review Count Over Time',fontsize=14)
plt.tick_params(labelsize=12)
plt.legend(title='',fontsize='12')
plt.tight_layout()
plt.show();


# The average length of negative reviews tends to be longer than positive reviews but the word count also varies considerably more.

# In[4]:


display(pd.DataFrame({'All': df['review_length'],
              'Positive' : df[df['classification']=='Positive']['review_length'],
              'Negative' : df[df['classification']=='Negative']['review_length']}).describe().drop(labels='count', axis=0).applymap(lambda x: f"{x:0.2f}"))

sns.boxplot(y = 'classification', 
    x = 'review_length', 
    data = df, 
    palette = ['#88D8B0', '#FF6F69'],
   showfliers = False)
plt.ylabel('')
plt.xlabel('Review Length (Words)')
plt.title('Distribution of Review Lengths\n(Outliers Suppressed)');


# The mean length of both positive and negative reviews has continued to vary since release.

# In[5]:


# plot mean review length over time
ax =sns.lineplot(x=df['date'].dt.strftime('%b %d'),
            y='review_length',
            palette = ['#88D8B0', '#FF6F69'],
            hue='classification',
            data=df,
            )

plt.xlabel("")
plt.xticks(rotation=45)
plt.ylabel("Mean Word Count",fontsize=14)
plt.legend(loc=0,fontsize='13')
plt.title("Review Length Over Time",fontsize=14)
plt.tick_params(labelsize=14)
plt.tight_layout();

