# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:32:18 2020

@author: mahes
Erstellt für jedes Jahr ein LDA Modell auf Basis von vorbearbeiteten Daten. 
Die Datei/Funktion ist obsolet, aufgrund der Tatsache das Gensim dafür ein Algorithmus anbietet 
und mit diesem nur ein Model entsteht. Unklar wäre, falls es genutzt werden sollte, 
wie die Modelle in ein Programm sinnvoll eingebracht werden.
"""
import pandas as pd 

import gensim
from gensim.models.phrases import Phraser
from gensim.models import CoherenceModel
import pyLDAvis.gensim
import locale
locale.setlocale(locale.LC_ALL, 'de_DE')
import dataProcessHelper as dph
from gensim.test.utils import common_corpus, common_dictionary

ldaPath = 'ldaModel/ldaModelTrigramYear/'


numberTopics = 8

path = dph.getDataPath('pressBiTriLemma.json')
dfBase = pd.read_json(path)
dfBase['time'] = pd.to_datetime(dfBase['time'], format="%A, %d. %B %Y") 

for year in dfBase['time'].dt.year.unique()[::-1]:
    print('\n------' + str(year) + '------')    
    
    df = dfBase.loc[dfBase['time'].dt.year == year]
    dfPreprocessed = df['trigramLemma']    
    bowCorpus, dictionary = dph.getCorpus(dfPreprocessed)
    
    ldaModel = gensim.models.LdaModel(bowCorpus, num_topics=numberTopics, random_state=100, id2word=dictionary, passes=1, iterations=100, per_word_topics=True)
    dph.saveModel(ldaModel, ldaPath + str(year))
   
    print('\nPerplexity:', ldaModel.log_perplexity(bowCorpus))
    
    cm = CoherenceModel(model=ldaModel, corpus=bowCorpus, coherence='u_mass')
    coherence = cm.get_coherence()  # get coherence value
    
    print('\nCoherence Score:', coherence)
    print()
    for idx, topic in ldaModel.print_topics(-1):
       print('Topic: {} \nWords: {}'.format(idx, topic))
    
    
if False:
    LDAvisPrepared = pyLDAvis.gensim.prepare(ldaModel, bowCorpus, dictionary)
    pyLDAvis.show(LDAvisPrepared)



    
