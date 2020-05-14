# -*- coding: utf-8 -*-
"""
Created on Fri May  8 02:03:30 2020

@author: mahes
"""
import string
import os
import pandas as pd
import numpy as np

#from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
#from nltk.corpus import stopwords
import de_core_news_sm
nlp = de_core_news_sm.load()
import pickle

import gensim

def isStopPunctSpace(token):
    return token.is_punct or token.is_space or token.is_stop or token.like_num

def combineStrings(df, row):
    return df["shortText"].iloc[row.name] + " " + removeImagePart(df['richText'].iloc[row.name])

def removeImagePart(richTextString):
    return richTextString.split('Foto:')[0] if len(richTextString) < 100 else richTextString

def saveAsJson(dfTemp, filename):
    with open(getDataPath(filename), 'w', encoding='utf-8') as file:
        dfTemp.to_json(file, force_ascii=False)
        
def saveModel(model, modelname):
    with open(getModelPath(modelname), 'wb') as f:
        pickle.dump(model, f)
        
def loadModel(modelname):
    with open(getModelPath(modelname), 'rb') as f:
        return pickle.load(f)

#Function without bi and trigram.
def createStopPunctLemma(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not isStopPunctSpace(token):    
            tokens.append(token.lemma_)
    return tokens

def getDataPath(filename):
    return os.path.join('data', filename)

def getModelPath(modelname):
    return os.path.join('model', modelname)

def removeStopPunctSpace(texts):
    doc = nlp(texts)
    tokens = []
    for token in doc:
        if not isStopPunctSpace(token):
            tokens.append(token.text)
    return tokens

def createBigrams(texts, bigramMod):
    bigramList = []
    for doc in texts:
        tempResult = bigramMod[doc]
        bigramList.append(tempResult)
    return bigramList

def createTrigrams(texts, bigramMod, trigramMod):
    trigramList = []
    for doc in texts:
        tempResult = trigramMod[bigramMod[doc]]
        trigramList.append(tempResult)
    return trigramList

def lemmatization(texts, allowedPostags):
    textsOut = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        textsOut.append([token.lemma_ for token in doc if token.pos_ in allowedPostags])
    return textsOut    

def getCorpus(data):
    dictionary = gensim.corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=8, no_above=0.2, keep_n=100000)
    dictionary.compactify()
    bowCorpus = [dictionary.doc2bow(doc) for doc in data]
    return bowCorpus, dictionary

def getNmfLdaTopic(model, vectorizer, numberTopics, nTopWords):
    term = vectorizer.get_feature_names()
    wordDict = {};
    for i in range(numberTopics):
        wordsIds = model.components_[i].argsort()[:-nTopWords - 1:-1]
        words = [term[key] for key in wordsIds]
        wordDict['Topic # ' + '{:02d}'.format(i+1)] = words;
    return pd.DataFrame(wordDict);

def getTopicsTermsWeights(weights, featureNames):
    featureNames = np.array(featureNames)
    sortedIndices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sortedWeights = np.array([list(wt[index]) for wt, index in zip(weights, sortedIndices)])
    sortedTerms = np.array([list(featureNames[row]) for row in sortedIndices])

    topics = [np.vstack((terms.T, termWeights.T)).T for terms, termWeights in zip(sortedTerms, sortedWeights)]

    return topics

def printTopicsUdf(topics, numberTopics=1, weightThreshold=0.0001, displayWeights=False, numTerms=None):
    for index in range(numberTopics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weightThreshold]
        if displayWeights:
            print('Topic #'+str(index+1)+' with weights')
            print(topic[:numTerms]) if numTerms else topic
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:numTerms]) if numTerms else tw
            
def getTopicsUdf(topics, numberTopics=1, weightThreshold=0.0001, numTerms=None):
    topicTerms = []
    for index in range(numberTopics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        topic = [(word, round(wt,2))
                 for word, wt in topic
                 if abs(wt) >= weightThreshold]
        topic_terms.append(topic[:numTerms] if numTerms else topic)
    return topicTerms