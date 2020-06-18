# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:01:02 2020

@author: mahes
In der Datei befindet sich eine Pipeline mit dem LDASeq Algorithmus. 
Übergeben werden eine vorbearbeitete Datenbasis und die Anzahl an Dokumente für jede Periode. 
Ausgeben wird ein Topic für jede Periode. Andere Ausgaben sind mögliche siehe Gensim-Dokumentation.
"""

import pandas as pd 

import gensim
from gensim.models.phrases import Phraser
from gensim.models import CoherenceModel
import pyLDAvis.gensim
from gensim.models import LdaSeqModel
import numpy as np
import dataProcessHelper as dph

from gensim.test.utils import common_corpus, common_dictionary

numberTopics = 10
columnName = 'trigramLemma'

path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)


dfLemma = df[columnName]


df['time'] = pd.to_datetime(df['time'], format="%A, %d. %B	%Y") 
uniqueYears, timeSlices = np.unique(df['time'].dt.year, return_counts=True) 

 
bowCorpus, dictionary = dph.getCorpus(dfLemma, noBelow=1, noAbove=0.2, keepN=2000)


if False:
    Model = gensim.models.LdaSeqModel(corpus=bowCorpus, num_topics=numberTopics, time_slice=timeSlices, chunksize=1, id2word=dictionary)
    dph.saveModel(Model, 'ldaseqModel' + columnName + str(numberTopics))
else:
    #ldaseqModelTrigram und ldaseqModelTrigram2
    Model = dph.loadModel('ldaseqModel' + columnName + str(numberTopics))
    

idx = 0
for topic in Model.print_topic_times(topic=1, top_terms=10):
    print('Time: {} \nWords: {}'.format(idx, topic))
    idx += 1
    