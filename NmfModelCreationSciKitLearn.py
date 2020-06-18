# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:30:13 2020

@author: mahes
Scikit-Learn Nmf Implementierung mit Themenausgabe und deren Gewichte. 
"""

import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;
from sklearn.decomposition import NMF;


import dataProcessHelper as dph

numberTopics = 15
columnName = "trigramLemma"

path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)


df = df[columnName]

df2 = pd.DataFrame(df)
df2[columnName] = df2.apply(lambda row: ' '.join(map(str, row[columnName])), axis=1)


#vectorizer = CountVectorizer(strip_accents = 'unicode');
#x_counts = vectorizer.fit_transform(df2['combCleanLemma']);
#transformer = TfidfTransformer();
#x_tfidf = transformer.fit_transform(x_counts);
#xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

vectorizer = TfidfVectorizer(strip_accents = 'unicode', ngram_range = (1,2));
xTrainTfidf = vectorizer.fit_transform(df2[columnName]);

if True:
    model = NMF(n_components=numberTopics, init='nndsvd');
    dph.saveModel(model, 'nmfModel' + 'columnName' + str(numberTopics))
else:
    model = dph.loadModel('nmfModel' + 'columnName' + str(numberTopics))

model = NMF(n_components=numberTopics, init='nndsvd');
model.fit(xTrainTfidf)

#Ermittelt die Werte für die Ausgaben von Topics und deren Gewichtung
nmfFeatureNames = vectorizer.get_feature_names()
nmfWeights = model.components_

#Gibt die Topics aus
result = dph.getNmfLdaTopic(model, vectorizer, numberTopics, 20)
print(result)

#Gibt die topics mit Wörtern und Gewichten aus.
topics = dph.getNmfLdaTopicsTermsWeights(nmfWeights, nmfFeatureNames)
dph.printTopicsUdf(topics, numberTopics, numTerms=10, displayWeights=True)

