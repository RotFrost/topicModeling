# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:30:13 2020

@author: mahes
"""

import pandas as pd 

import dataProcessHelper as dph
import pyLDAvis.sklearn

numberTopics = 10

path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)


dfBigramLemma = df['combCleanLemma']
#dfCombClean = dfCombClean.iloc[-100:]

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;

df2 = pd.DataFrame(dfBigramLemma)
df2['combCleanLemma'] = df2.apply(lambda row: ' '.join(map(str, row.combCleanLemma)), axis=1)


vectorizer = CountVectorizer(strip_accents = 'unicode');
x_counts = vectorizer.fit_transform(df2['combCleanLemma']);
transformer = TfidfTransformer();
x_tfidf = transformer.fit_transform(x_counts);
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

model = NMF(n_components=numberTopics, init='nndsvd');
model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(numberTopics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);

resultDict = get_nmf_topics(model, 20)

