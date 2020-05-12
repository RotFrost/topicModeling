# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:06:42 2020

@author: mahes
"""

import pandas as pd

def formatTopicsSentences(corpus, texts, ldaModel=None):
    # Init output
    dfSentTopics = pd.DataFrame()

    # Get main topic in each document
    for i, rowList in enumerate(ldamodel[corpus]):
        row = rowList[0] if ldamodel.per_word_topics else rowList            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topicNum, propTopic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topicNum)
                topicKeywords = ", ".join([word for word, prop in wp])
                dfSentTopics = dfSentTopics.append(pd.Series([int(topicNum), round(propTopic,4), topicKeywords]), ignore_index=True)
            else:
                break
    dfSentTopics.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    dfSentTopics = pd.concat([dfSentTopics, contents], axis=1)
    return(dfSentTopics)