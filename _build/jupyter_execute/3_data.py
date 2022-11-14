#!/usr/bin/env python
# coding: utf-8

# # The Data

# ```{figure} https://cdn.akamai.steamstatic.com/steam/apps/1938090/header.jpg?t=1668017465
# ---
# align: center
# ---
# ```
# 
# Review data for the title ['Call of Duty: Modern Warfare 2'](https://store.steampowered.com/app/1938090/Call_of_Duty_Modern_Warfare_II/) published by Activision were collected. 
# 
# At the time of access (2022-11-14), this title held a 'Mixed' review score based on 68,626 user reviews.
# 
# The Steam store uses a binary review classification system in which users can 'recommend' or 'not recommend' a title. Many titles display severely skewed review classifications which would generate an extremely imbalanced sample. The 'Mixed' classification of this title indicates a more even split between the possible review classifications.
# 
# Reviews were scraped from the Steam store using the `steamreviews` API for Python {cite}`wok_2018`.

# In[1]:


# api access
import steamreviews

# set parameters
request_params = dict()
request_params['language'] = 'english'
request_params['purchase_type'] = 'all'
app_id = 1938090

# store results as dictionary
review_dict, query_count = steamreviews.download_reviews_for_app_id(app_id,chosen_request_params=request_params)


# All available English language reviews were scraped, forming an initial sample of 47,356 observations.     
# 3 features were extracted, including:
# 
# - Review date
# - Review text
# - Review classification
# 
# An an additional feature, `review_length`, calculates the number of words in the review text and was added to the set.

# In[2]:


import pandas as pd

review_id = [x for x in review_dict['reviews']]
date = [review_dict['reviews'][x]['timestamp_created'] for x in review_id]
review_text = [review_dict['reviews'][x]['review'] for x in review_id]
classification = [review_dict['reviews'][x]['voted_up'] for x in review_id]


df = pd.DataFrame(list(zip(date,review_text,classification)),
                 columns=['date','review_text','classification'])

# calculate review text length, set as feature
df['review_length'] = df['review_text'].str.split().str.len().fillna(0)

df


# ## Inital Clean-up
# 
# Prior to conducting any exploratory analysis, some basic cleaning was performed:
# 
# 1. Replace boolean values for the `classification` (voted_up) variable with strings ('Positive', 'Negative')
# 2. Convert unix timestamp in `date` to datetime (YYYY-MM-DD)
# 3. Drop all entries with missing review text
# 
# The resulting data frame is composed of 47,191 observations and is stored as a .csv for use in subsequent stages of the analysis.

# In[3]:


import numpy as np
from datetime import datetime

# replace boolean values with strings
df['classification'].replace([True,False],['Positive','Negative'],inplace=True)

# convert unix time stamp to datetime64
df['date'] = pd.to_datetime(df['date'], unit='s').dt.normalize()

# Keep reviews with >=1 word
df = df.drop(df[df['review_text'].str.split().str.len()<1].index)

df.to_csv('data/processed_review_data.csv',index=False)

df

