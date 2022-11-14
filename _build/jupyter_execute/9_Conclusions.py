#!/usr/bin/env python
# coding: utf-8

# # Conclusions
# 
# ![negtokens](images/postokens.png)

# Looking at the extracted noun tokens for each classification, it is clear that not much insight into the relevant aspects of game design can be inferred from the results.
# 
# It would seem that the method of extracting the strongest coefficients from the fitted model is not an effective means of identifying salient issues.
# 
# The likely cause of this failing is that the method employed here does not match identified aspects of game design referenced in the text with associated sentiment. Rather, this method unsuccessfully attempts to derive an understanding of sentiment from the strength of a token's coefficient in each classification. The logic of this is particularly flawed given the nature of Steam review content. While a review may be categorised as 'not recommend', the contents of that review is rarely all negative. Steam reviews have a tendency to include a variety of both positive and negative points yet this method is unable to distinguish those within each review.
# 
# Greater success in identifying prominent issues of game design will likely be experienced by performing aspect based sentiment analysis. This would enable a sentence by sentence analysis of the review text and the assignment of a polarity score per sentence as opposed to an overall classification for each review. In doing so, this is far more likely to provide an accurate indiciation of specific positive or negative aspects of game design contained within the text of each user eview. This approach will, therefore, form the basis of follow-up work.
# 
# Despite this, notable success was demonstrated in the accurate classification of user reviews on the basis of included text. Small victory!
# 
# ![negtokens](images/negtokens.png)
# 
