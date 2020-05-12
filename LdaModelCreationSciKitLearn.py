# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:08:17 2020

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

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;
from sklearn.decomposition import LatentDirichletAllocation as LDA;
from sklearn.preprocessing import normalize;
import pickle;

df2 = pd.DataFrame(dfBigramLemma)
df2['combCleanLemma'] = df2.apply(lambda row: ' '.join(map(str, row.combCleanLemma)), axis=1)


vectorizer = TfidfVectorizer(strip_accents = 'unicode', ngram_range = (1,2));
x_counts = vectorizer.fit_transform(df2['combCleanLemma']);
#transformer = TfidfTransformer();
#x_tfidf = transformer.fit_transform(x_counts);
#xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

model = LDA(n_components=numberTopics);
model.fit(x_counts)

def get_nmf_topics(model, n_top_words):
    
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {};
    for i in range(numberTopics):
        
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);

resultDict = get_nmf_topics(model, 20)

LDAvisPrepared = pyLDAvis.sklearn.prepare(model, x_counts, vectorizer)
pyLDAvis.show(LDAvisPrepared)