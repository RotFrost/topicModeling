# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:25:49 2020

@author: Benjamin
"""

import pandas as pd 

import gensim
from gensim.models import LdaSeqModel #seq LDA of Blei Lafferty
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

print(len(bowCorpus))


if True:
    #Model = gensim.models.HdpModel(bowCorpus, id2word=dictionary)
    Model = gensim.models.LdaSeqModel(corpus=bowCorpus, num_topics=numberTopics, chunksize=1, id2word=dictionary)
    dph.saveModel(Model, 'ldaseqModelTrigram')
else:
    Model = dph.loadModel('ldaseqModelTrigram')
    
    
for idx, topic in Model.print_topics():
    print('Topic: {} \nWords: {}'.format(idx, topic))
    


#Choice of parameters

#lda    
    #One can choose different LDA Models as parameter lda = . The input is then an instance from the lda class.

#alphas
    #Also you can define the alphas, that is the priors
    
#time_slice    
    #the sum of time_slice must add up to the number of documents in the corpus. Can we get the number of documents in the corpus? 
    #How to compute the length of a corpus?
