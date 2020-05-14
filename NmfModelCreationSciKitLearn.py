# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:30:13 2020

@author: mahes
"""

import pandas as pd 

import matplotlib.pyplot as plt 

import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;

import dataProcessHelper as dph

numberTopics = 10

path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)


dfBigramLemma = df['combCleanLemma']
#dfCombClean = dfCombClean.iloc[-100:]

df2 = pd.DataFrame(dfBigramLemma)
df2['combCleanLemma'] = df2.apply(lambda row: ' '.join(map(str, row.combCleanLemma)), axis=1)


#vectorizer = CountVectorizer(strip_accents = 'unicode');
#x_counts = vectorizer.fit_transform(df2['combCleanLemma']);
#transformer = TfidfTransformer();
#x_tfidf = transformer.fit_transform(x_counts);
#xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

vectorizer = TfidfVectorizer(strip_accents = 'unicode', ngram_range = (1,2));
xTrainTfidf = vectorizer.fit_transform(df2['combCleanLemma']);

model = NMF(n_components=numberTopics, init='nndsvd');
model.fit(xTrainTfidf)

nmfFeatureNames = vectorizer.get_feature_names()
nmfWeights = model.components_

resultDict = dph.getNmfLdaTopic(model, vectorizer, numberTopics, 20)
print(resultDict)

topics = dph.getTopicsTermsWeights(nmfWeights, nmfFeatureNames)
dph.printTopicsUdf(topics, numberTopics, numTerms=10, displayWeights=True)