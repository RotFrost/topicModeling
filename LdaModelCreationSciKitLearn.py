# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:08:17 2020

@author: mahes
"""

import pandas as pd 

import dataProcessHelper as dph
import pyLDAvis.sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;
from sklearn.decomposition import LatentDirichletAllocation as LDA;
from sklearn.preprocessing import normalize;
import pickle;

from sklearn.model_selection import GridSearchCV


numberTopics = 10

path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)


dfBigramLemma = df['combCleanLemma']
#dfCombClean = dfCombClean.iloc[-100:]


df2 = pd.DataFrame(dfBigramLemma)
df2['combCleanLemma'] = df2.apply(lambda row: ' '.join(map(str, row.combCleanLemma)), axis=1)


vectorizer = TfidfVectorizer(strip_accents = 'unicode', ngram_range = (1,2));
xTrainTfidf = vectorizer.fit_transform(df2['combCleanLemma']);
#transformer = TfidfTransformer();
#x_tfidf = transformer.fit_transform(x_counts);
#xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

searchParams = {'n_components': [10, 15, 20, 25], 'learning_decay': [.5, .7, .9]}

model = LDA()

model = GridSearchCV(model, searchParams)
model.fit(xTrainTfidf)

model = model.best_estimator_
print("Best model's params: ", model.best_params_)
print("Best log likelihood score: ", model.best_score_)
print("Model perplexity: ", model.perplexity(xTrainTfidf))


nmfFeatureNames = vectorizer.get_feature_names()
nmfWeights = model.components_

resultDict = dph.getNmfLdaTopic(model, vectorizer, numberTopics, 20)
print(resultDict)

topics = dph.getTopicsTermsWeights(nmfWeights, nmfFeatureNames)
dph.printTopicsUdf(topics, numberTopics, numTerms=10, displayWeights=True)

LDAvisPrepared = pyLDAvis.sklearn.prepare(model, xTrainTfidf, vectorizer)
pyLDAvis.show(LDAvisPrepared)

