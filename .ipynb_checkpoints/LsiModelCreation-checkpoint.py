# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:34:36 2020

@author: mahes
"""

import pandas as pd 

import gensim
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
    lsiModel = gensim.models.LsiModel(bowCorpus, num_topics=numberTopics, id2word=dictionary)
    dph.saveModel(lsiModel, 'LsiModelTrigram')
else:
    lsiModel = dph.loadModel('LsiModelTrigram')
    
for idx, topic in lsiModel.print_topics(-1):
   print('Topic: {} \nWords: {}'.format(idx, topic))

#cm = CoherenceModel(model=ldaModel, corpus=bowCorpus, coherence='u_mass')
#coherence = cm.get_coherence()  # get coherence value


#print('\nCoherence Score:', coherence)


