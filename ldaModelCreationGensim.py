# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:32:18 2020

@author: mahes
Einfache Implementierung des Algorithmus „Latent Dirichlet Allocation“, welche auf ein zuvor bereinigte Datenbasis zugreift. 
Gibt den Coherence-Score (c_v) und die Perplexity aus sowie die Topics. 
Das Laden von gespeicherten Modellen ist möglich. 
Kann außerdem drei Visualisierungen ausgeben für den Perplexity, Coherence-Score (u_mass, c_v) über die Anzahl der Topics.
"""
import pandas as pd 

import gensim
from gensim.models import CoherenceModel
import pyLDAvis.gensim
import locale
import topicVisualization as tV
locale.setlocale(locale.LC_ALL, 'de_DE')

import dataProcessHelper as dph

#Notwendig um nicht auf ein BrokenPipeError zu stoßen auf Windows
if __name__ == "__main__":
    path = 'ldaModel'
    
    numberTopics = 6
    
    path = dph.getDataPath('pressBiTriLemma.json')
    df = pd.read_json(path)
    dfPre = df['trigramLemma']
       
    bowCorpus, dictionary = dph.getCorpus(dfPre)
    texts = [[dictionary[word_id] for word_id, freq in doc] for doc in bowCorpus]
    
    #Falls tf-idf verwendet werden soll.
    if True:
        tfidf = gensim.models.TfidfModel(bowCorpus)
        bowCorpus = tfidf[bowCorpus]
    
    
    if True:
        ldaModel = gensim.models.LdaModel(bowCorpus, num_topics=numberTopics, alpha=0.1, random_state=130, id2word=dictionary, passes=20, iterations=100, per_word_topics=True)
        if False:
            dph.saveModel(ldaModel, 'ldaModel')
    else:
       ldaModel = dph.loadModel('ldaModel')   
    
    #Zeigt die Topics und den Score an.
    if True:
        print('\nPerplexity:', ldaModel.log_perplexity(bowCorpus))
    
        cm = CoherenceModel(model=ldaModel, texts=texts, dictionary=dictionary, corpus=bowCorpus, coherence='c_v')
        coherence = cm.get_coherence() 
        
        print('\nCoherence Score:', coherence)
        
        for idx, topic in ldaModel.print_topics(-1):
           print('Topic: {} \nWords: {}'.format(idx, topic))
        
    #Zeigt LdaVis
    if False:
        LDAvisPrepared = pyLDAvis.gensim.prepare(ldaModel, bowCorpus, dictionary)
        pyLDAvis.show(LDAvisPrepared)
    
    #Zeigt drei Plots mit den Coherence Scores und perplexity über die Anzahl der Topics
    if True:
        tV.plotGraphTopicNForLsiLda(dfPre,'ldaModel', 30)
        
# # Code erstellt den texts input für die Berechnung des coherence c_v score 
# dfPre = df['trigramLemma']
# texts = dfPre.to_numpy()