# -*- coding: utf-8 -*-
"""
Created on Fri May  8 02:03:30 2020

@author: mahes
"""
import string
import os
import pandas as pd

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