#!/usr/bin/env python
# coding: utf-8

# # Conclusions
# 
# ![negtokens](images/postokens.png)

# Looking at the extracted noun tokens for each classification, it is clear that not much insight into the relevant aspects of game design can be inferred from the results.
# 
# It would seem that the method of extracting the strongest coefficients from the fitted model is not an effective means of identifying salient issues.
# 
# The likely cause of this failing is that the method employed here does not match identified aspects of game design referenced in the text with associated sentiment. Rather, this method unsuccessfully attempts to derive an understanding of sentiment from the strength of a token's coefficient in each classification.
# 
# Greater success in identifying prominent issues of game design will likely be experienced from an aspect based sentiment analysis approach which will form the basis of follow-up work.
# 
# Despite this, notable success was demonstrated in the accurate classification of user reviews on the basis of included text.
# 

# <p float="left">
#   <img src="images/postokens.png" width="480" />
#   <img src="images/negtokens.png" width="480" /> 
# </p>
