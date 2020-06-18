# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:07:41 2020

@author: mahes

Einfache Implementierung des Algorithmus „k-Means“, welche auf ein zuvor bereinigte Datenbasis zugreift und 
dann mittels dem Tf-idf-Vectorizer für den Algorithmus zugänglich macht. 
Gibt die Topics aus und erstellt zwei Plots einen ohne Label und deren anderen mit Label.
Gespeicherte Modelle können wieder geladen werden.
Zum testen/ausprobieren.
"""
import pandas as pd

import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer;
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
import dataProcessHelper as dph
from sklearn.decomposition import TruncatedSVD as svd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

modelType = 'kmeans'

numberTopics = 13
columnName = "trigramLemma"

path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)


dfpre = df[columnName]
#dfCombClean = dfCombClean.iloc[-100:]


df2 = pd.DataFrame(dfpre)
df2[columnName] = df2.apply(lambda row: ' '.join(map(str, row[columnName])), axis=1)


vectorizer = TfidfVectorizer(strip_accents = 'unicode', ngram_range = (1,2));
xTrainTfidf = vectorizer.fit_transform(df2[columnName]);

if True:
    model = KMeans(n_clusters=numberTopics, init='k-means++', max_iter=100, n_init=1)
    model.fit(xTrainTfidf)
    if False:
        dph.saveModel(model, modelType + columnName + str(numberTopics))
else:
    model = dph.loadModel(modelType + columnName + str(numberTopics))


orderCentroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(numberTopics):
    print("Cluster %d:" % i, end='')
    for ind in orderCentroids[i, :15]:
        print(' %s' % terms[ind], end='')
    print('\n')
    

labels = model.predict(xTrainTfidf)

data2d_old = svd(n_components=2)
data2D = data2d_old.fit_transform(xTrainTfidf)

plt.scatter(data2D[:,0], data2D[:,1])
plt.show() 

plt.title('K-Means result')
plt.scatter(data2D[:,0], data2D[:,1], c = labels)
plt.show() 

rangeClusters = [2, 3, 4, 5, 6]

'''
Code führt zum Absturz

X = xTrainTfidf.todense()
clusterLabels = model.fit_predict(xTrainTfidf)
silhouetteAvg = silhouette_score(X, clusterLabels)

print(silhouettenAvg)
'''
