# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:06:42 2020

@author: mahes
"""

import pandas as pd
import gensim
from gensim.models import CoherenceModel
import dataProcessHelper as dph
import matplotlib.pyplot as plt

def formatTopicsSentences(corpus, texts, ldaModel=None):
    dfSentTopics = pd.DataFrame()

    for i, rowList in enumerate(ldaModel[corpus]):
        row = rowList[0] if ldaModel.per_word_topics else rowList            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topicNum, propTopic) in enumerate(row):
            if j == 0: 
                wp = ldaModel.show_topic(topicNum)
                topicKeywords = ", ".join([word for word, prop in wp])
                dfSentTopics = dfSentTopics.append(pd.Series([int(topicNum), round(propTopic,4), topicKeywords]), ignore_index=True)
            else:
                break
    dfSentTopics.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    contents = pd.Series(texts)
    dfSentTopics = pd.concat([dfSentTopics, contents], axis=1)

    return(dfSentTopics)


def plotGraphTopicNForLsiLda(dfClean, modelType, stop, start=2, step=1):
    bowCorpus, dictionary = dph.getCorpus(dfClean)
    texts = [[dictionary[word_id] for word_id, freq in doc] for doc in bowCorpus]
    modelList, coherenceValues, perplexityValues, coherenceCVValues = computeCoherenceValues(dictionary, bowCorpus, texts, modelType, stop, start, step)

    x = range(start, stop, step)
    plt.plot(x, coherenceValues)
    plt.title(modelType)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherenceValues"), loc='best')
    plt.show()
    
    x = range(start, stop, step)
    plt.plot(x, coherenceCVValues)
    plt.title(modelType)
    plt.xlabel("Number of Topics")
    plt.ylabel("CoherenceCV score")
    plt.legend(("coherenceCVValues"), loc='best')
    plt.show()
    
    if (modelType == 'ldaModel'):
        x = range(start, stop, step)
        plt.plot(x, perplexityValues)
        plt.title(modelType)
        plt.xlabel("Number of Topics")
        plt.ylabel("Perplexity score")
        plt.legend(("perlexityValues"), loc='best')
        plt.show()

def computeCoherenceValues(dictionary, bowCorpus, texts, modelType, stop, start=2, step=1):
    perplexityValues = []
    coherenceValues = []
    coherenceCVValues = []
    modelList = []
    for topicN in range(start, stop, step):
        if (modelType == 'lsiModel'):
            model = gensim.models.LsiModel(bowCorpus, num_topics=topicN, id2word = dictionary)
        else:
            model = gensim.models.LdaModel(bowCorpus, num_topics=topicN, id2word=dictionary, random_state=100)
            perplexityValues.append(model.log_perplexity(bowCorpus))
            
        modelList.append(model)
        coherencemodel = CoherenceModel(model=model, corpus = bowCorpus, dictionary=dictionary, coherence='u_mass')
        coherenceCVModel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, corpus=bowCorpus, coherence='c_v')
        coherenceCVValues.append(coherenceCVModel.get_coherence())
        
        coherenceValues.append(coherencemodel.get_coherence())
    return modelList, coherenceValues, perplexityValues, coherenceCVValues

def plotGraphTopicNForLdaPerplexity(dfClean, stop, start=2, step=1):
    bowCorpus, dictionary = dph.getCorpus(dfClean)
    
    modelList, perplexityValues = computePerplexityValues(dictionary, bowCorpus, dfClean,stop, start, step)

    x = range(start, stop, step)
    plt.plot(x, perplexityValues)
    plt.title(modelType)
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity score")
    plt.legend(("perlexityValues"), loc='best')
    plt.show()
    
def computePerplexityValues(dictionary, bowCorpus, docClean, modelType, stop, start=2, step=1):
    perplexityValues = []
    modelList = []
    for topicN in range(start, stop, step):
        ldaModel = gensim.models.LdaModel(bowCorpus, num_topics=topicN, id2word=dictionary, random_state=100)
        modelList.append(ldaModel)
        perplexityValues.append(ldaModel.log_perplexity(bowCorpus))
        
    return modelList, perplexityValues