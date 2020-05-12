# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:25:49 2020

@author: mahes
"""

import pandas as pd 

import gensim
from gensim.models import HdpModel
from gensim.models.phrases import Phraser
from gensim.models import CoherenceModel
import pyLDAvis.gensim

import dataProcessHelper as dph

from gensim.test.utils import common_corpus, common_dictionary

numberTopics = 10

path = dph.getDataPath('pressBiTriLemma.json')
df = pd.read_json(path)

dfBigramLemma = df['trigramLemma']
#dfCombClean = dfCombClean.iloc[-100:]
 
bowCorpus, dictionary = dph.getCorpus(dfBigramLemma)

if True:
    Model = gensim.models.HdpModel(bowCorpus, id2word=dictionary)
    dph.saveModel(Model, 'hdpModelTrigram')
else:
    Model = dph.loadModel('hdpModelTrigram')
    
    
for idx, topic in Model.print_topics():
    print('Topic: {} \nWords: {}'.format(idx, topic))