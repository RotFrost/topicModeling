# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:25:49 2020

@author: mahes

Einfache Implementierung des Algorithmus „Hierarchical Dirichlet Process“, welche auf ein zuvor bereinigte Datenbasis zugreift.
Gibt den Coherence-Score (u_mass) aus sowie die Anzahl der Topics als auch die Topics selbst. Modelle können wieder geladen werden.
Zum testen/ausprobieren.
"""

import pandas as pd 

import gensim
from gensim.models import CoherenceModel

import dataProcessHelper as dph


columnName = 'trigramLemma'
path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)

dfBigramLemma = df[columnName]
#dfCombClean = dfCombClean.iloc[-100:]
 
bowCorpus, dictionary = dph.getCorpus(dfBigramLemma)

if True:
    hdpModel = gensim.models.HdpModel(bowCorpus, id2word=dictionary,
                                      max_chunks=None, max_time=None, chunksize=256, kappa=1.0, 
                                      tau=64.0, K=15, T=150, alpha=1, gamma=1, eta=0.01, scale=1.0, 
                                      var_converge=0.1, outputdir=None, random_state=None)
    if False:
        dph.saveModel(hdpModel, 'hdpModel' + columnName)
else:
    hdpModel = dph.loadModel('hdpModel' + columnName)

cm = CoherenceModel(model=hdpModel, corpus=bowCorpus, coherence='u_mass')
coherence = cm.get_coherence()  # get coherence value

topicMatrix = hdpModel.get_topics()
print('Topic number: ', topicMatrix.shape[0])

print('\nCoherence Score:', coherence)
    
for idx, topic in hdpModel.print_topics(num_topics=30):
    print('Topic: {} \nWords: {}'.format(idx, topic))