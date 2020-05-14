# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:32:18 2020

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


numberTopicsList = [5,10,15,20]
resultDict = {}

for numberTopic in numberTopicsList:
    ldaModel = gensim.models.LdaModel(bowCorpus, num_topics=numberTopics, random_state=100, id2word=dictionary, passes=1, iterations=100, per_word_topics=True)
    cm = CoherenceModel(model=ldaModel, corpus=bowCorpus, coherence='u_mass')
    coherence = cm.get_coherence()
    resultDict[numberTopic] = coherence
    
print(resultDict)
    
    
if True:
    ldaModel = gensim.models.LdaModel(bowCorpus, num_topics=numberTopics, random_state=100, id2word=dictionary, passes=1, iterations=100, per_word_topics=True)
    dph.saveModel(ldaModel, 'ldaModelTrigram')
else:
   ldaModel = dph.loadModel('ldaModelTrigram')
    
    
for idx, topic in ldaModel.print_topics(-1):
   print('Topic: {} \nWords: {}'.format(idx, topic))
    
    
print('\nPerplexity:', ldaModel.log_perplexity(bowCorpus))

cm = CoherenceModel(model=ldaModel, corpus=bowCorpus, coherence='u_mass')
coherence = cm.get_coherence()  # get coherence value


print('\nCoherence Score:', coherence)

for idx, topic in ldaModel.print_topics(-1):
   print('Topic: {} \nWords: {}'.format(idx, topic))
    
if False:
    LDAvisPrepared = pyLDAvis.gensim.prepare(ldaModel, bowCorpus, dictionary)
    pyLDAvis.show(LDAvisPrepared)

train_vecs = []
for i in range(len(df)):
    top_topics = ldaModel.get_document_topics(bowCorpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(numberTopics)]
    train_vecs.append(topic_vec)
    
