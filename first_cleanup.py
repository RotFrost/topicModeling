# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:59:00 2020

@author: mahes
"""
import string
import os
import pandas as pd

#from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
#from nltk.corpus import stopwords

import spacy
import de_core_news_sm
nlp = de_core_news_sm.load()

import gensim

import pickle

def isStopPunctSpace(token):
    return token.is_punct or token.is_space or token.is_stop

def combineStrings(df, row):
    return df["shortText"].iloc[row.name] + " " + removeImagePart(df['richText'].iloc[row.name])

def removeImagePart(richTextString):
    return richTextString.split('Foto:')[0] if len(richTextString) < 100 else richTextString

def saveAsJson(dfTemp, filename):
    with open(getDataPath(filename), 'w', encoding='utf-8') as file:
        dfTemp.to_json(file, force_ascii=False)
        
def saveModel(model, modelname):
    with open(getModelPath(modelname), 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
def loadModel(modelname):
    with open(getModelPath(modelname), 'rb') as f:
        return pickle.load(f)
        
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

richTextFilter = ['Zur externen Meldung', 'Zur externen Pressemitteilung']


#Combine the columns 'shortText' and 'richText'. What is with the title?
if False:
    path = getDataPath('press42.json')
    df = pd.read_json(path)
    df['combText'] = df.apply(lambda row: df['shortText'].iloc[row.name] if df['richText'].iloc[row.name] in richTextFilter else combineStrings(df, row), axis=1)
    saveAsJson(df, 'press42Combined.json')
#Load the combined data
if False:
    path = getDataPath('press42Combined.json')
    df = pd.read_json(path)

#Filter the comb with stopwords, punctuation and lemma or load the combined cleaned data
if False:
    df['combClean'] = df.apply(lambda row: createStopPunctLemma(df['combText'].iloc[row.name]), axis=1)
    saveAsJson(df, 'press42CombClean.json')
else:
    path = getDataPath('press42CombClean.json')
    df = pd.read_json(path)
    
dfCombClean = df['combClean']
    
dictionary = gensim.corpora.Dictionary(dfCombClean)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in dfCombClean]

if False:
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    saveModel(lda_model, 'lda_model')
else:
    lda_model = loadModel('lda_model')
    
    
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


#Stop words nltk
'''
stopWords = set(stopwords.words('german'))
words = word_tokenize(exampleText)
words = [word.lower() for word in words if word.isalpha()]
wordsFiltered = []

print(words)

for word in words:
    if word not in stopWords:
        wordsFiltered.append(word)

print(wordsFiltered)
'''

