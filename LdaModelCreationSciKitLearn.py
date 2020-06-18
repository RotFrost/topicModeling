# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:08:17 2020

@author: mahes

War eine einfache Scikit-Learn LDA Implementierung und wurde dann zu einem Versuch Gridsearch zur Parametersuche zu verwenden. 
Ausgegeben werden die Perplexity, und die Topics. 
Zum testen/ausprobieren. --- Funktioniert zwar die Ergebnisse sind nicht brauchbar.
"""

import pandas as pd 

import dataProcessHelper as dph
import pyLDAvis.sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;
from sklearn.decomposition import LatentDirichletAllocation as LDA;

from sklearn.preprocessing import normalize;
import pickle;
from sklearn.model_selection import GridSearchCV

import dataProcessHelper as dph

numberTopics = 10
columnName = 'combCleanLemma'

path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)

dfBigramLemma = df[columnName]

df2 = pd.DataFrame(dfBigramLemma)
df2[columnName] = df2.apply(lambda row: ' '.join(map(str, row[columnName])), axis=1)


vectorizer = TfidfVectorizer(strip_accents = 'unicode', ngram_range = (1,2));
xTrainTfidf = vectorizer.fit_transform(df2[columnName]);

searchParams = {'n_components': [10], 'learning_decay': [.5]}
if True:
    model = LDA()
    model = GridSearchCV(model, searchParams)
    model.fit(xTrainTfidf)
    model = model.best_estimator_
    if False:
        dph.saveModel(model, 'ldaGrid' + columnName)
else:
    model = dph.loadModel('ldaGrid' + columnName)

# Zeigt den Socre
print("Model perplexity: ", model.perplexity(xTrainTfidf))

#   Ermittelt die Werte für die nächsten Funktionen
featureNames = vectorizer.get_feature_names()
weights = model.components_

#Gibt die topics aus
result = dph.getNmfLdaTopic(model, vectorizer, numberTopics, 20)
print(result)

#Gibt die Topics und die Gewichte aus
topics = dph.getNmfLdaTopicsTermsWeights(weights, featureNames)
dph.printTopicsUdf(topics, numberTopics, numTerms=10, displayWeights=True)

#   Wendet ldaVis an
LDAvisPrepared = pyLDAvis.sklearn.prepare(model, xTrainTfidf, vectorizer)
pyLDAvis.show(LDAvisPrepared)

