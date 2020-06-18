# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:34:36 2020

@author: mahes
Einfache Implementierung des Latent Semantic Analysis Algorithmus. 
Ausgegeben werden der Coherence-Score und die Topics sowie zwei Visualisierungen 
jeweils für die zwei Coherence-Scores (u_mass, c_v) über die Anzahl der Topics.
"""
import pandas as pd 

import gensim
from gensim.models import CoherenceModel

import pyLDAvis.gensim

import dataProcessHelper as dph

import matplotlib.pyplot as plt
import topicVisualization as tV
    
if __name__ == "__main__":
    
    numberTopics = 15
    columnName = 'trigramLemma'
    
    path = dph.getDataPath('pressBiTriLemma.json')
    df = pd.read_json(path)
    
    dfPre = df[columnName]
    #dfCombClean = dfCombClean.iloc[-100:]
     
    bowCorpus, dictionary = dph.getCorpus(dfPre)
    
    if True:
        lsiModel = gensim.models.LsiModel(bowCorpus, num_topics=numberTopics, id2word=dictionary)
        dph.saveModel(lsiModel, 'lsiModel' + columnName + str(numberTopics))
    else:
        lsiModel = dph.loadModel('lsiModel' + columnName + str(numberTopics))
        
    for idx, topic in lsiModel.print_topics(-1):
       print('Topic: {} \nWords: {}'.format(idx, topic))
       
    cm = CoherenceModel(model=lsiModel, corpus=bowCorpus, coherence='u_mass')
    coherence = cm.get_coherence()  # get coherence value
    
    if True:
        texts = [[dictionary[word_id] for word_id, freq in doc] for doc in bowCorpus]
        cv = CoherenceModel(model=lsiModel, texts=texts, dictionary=dictionary, corpus=bowCorpus, coherence='c_v')
        coherenceCV = cv.get_coherence()
        print('\nCoherenceCV Score:', coherenceCV)
    
    print('\nCoherence Score:', coherence)
    
    if False:
        tV.plotGraphTopicNForLsiLda(dfPre,'lsiModel', 12)