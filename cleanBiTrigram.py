# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:59:00 2020

@author: mahes
"""
import pandas as pd
import de_core_news_sm
nlp = de_core_news_sm.load()

import gensim

import dataProcessHelper as dph

richTextFilter = ['Zur externen Meldung', 'Zur externen Pressemitteilung']
allowedPostags=['NOUN', 'ADJ', 'VERB', 'ADV']
#'Deutschland', 'Bundesregierung', 'wichtig', 'erklärte'

#Combine the columns 'shortText' and 'richText'. What is with the title?
if True:
    path = dph.getDataPath('press42.json')
    df = pd.read_json(path)
    df['combText'] = df.apply(lambda row: df['shortText'].iloc[row.name] if df['richText'].iloc[row.name] in richTextFilter else dph.combineStrings(df, row), axis=1)
    dph.saveAsJson(df, 'press42Combined.json')
#Load the combined data
if True:
    path = dph.getDataPath('press42Combined.json')
    df = pd.read_json(path)

if True:
    df['combClean'] = df.apply(lambda row: dph.removeStopPunctSpace(df['combText'].iloc[row.name]), axis=1)
    df['combCleanLemma'] = df.apply(lambda row: dph.createStopPunctLemma(df['combText'].iloc[row.name]), axis=1)
    dph.saveAsJson(df, 'press42CombClean.json')
else:
    path = dph.getDataPath('press42CombClean.json')
    df = pd.read_json(path)

if True:
    bigram = gensim.models.Phrases(df['combText'], min_count=1, threshold=0.8)
    trigram = gensim.models.Phrases(bigram[df['combText']], threshold=0.8)  
    
    bigramMod = gensim.models.phrases.Phraser(bigram)
    trigramMod = gensim.models.phrases.Phraser(trigram)
    
    dataWordsBigram = dph.createBigrams(df['combClean'], bigramMod)
    dataWordsTrigram = dph.createTrigrams(df['combClean'], bigramMod, trigramMod)
    
    df['bigramLemma'] = dph.lemmatization(dataWordsBigram, allowedPostags)
    df['trigramLemma'] = dph.lemmatization(dataWordsTrigram, allowedPostags)
    
    dph.saveAsJson(df, 'pressBiTriLemma.json')
    
if False:
    path = dph.getDataPath('pressBiTriLemma.json')
    df = pd.read_json(path)


