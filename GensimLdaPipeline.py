# -*- coding: utf-8 -*-
'''
Diese sehr ähnlich aufgebauten Dateien bilden die Pipeline für die Gensim-Algorithmen Hdp, Lda und Lsi. 
Im ersten Schritt ist es stets nötigt die folgenden Dictionaries SettingsDict, coprusDict und extraDict zu definieren, 
da diese Parameter für Funktionen beinhalten. Nachdem die Data-Preparation abgeschlossen wurde, wird das jeweilige Model trainiert. 
Danach erfolgt die Validierung mit dem Coherence-Score (u_mass und c_v) und wenn es sich um LDA handelt auch mit der Perplexity. 
Die Ergebnisse werden in einem weiteren Dictionary gespeichert und an das bestehende resultRecordDict.json angehängt. 
Das resultRecordDict.json beinhaltet die vorher gewonnen Ergebnisse und Parameter.
'''

import pandas as pd
import de_core_news_sm
nlp = de_core_news_sm.load()
import locale
locale.setlocale(locale.LC_ALL, 'de_DE')
import gensim
import dataProcessHelper as dph
import uuid
import os
from gensim.models import CoherenceModel
import pyLDAvis.gensim

#Notwendig um nicht auf ein BrokenPipeError zu stoßen auf Windows
if __name__ == "__main__":
    modelType = 'ldaGensim'
    numberTopics = 40
    
    #SettingsDict params:
    #biLemma, biStemm, triLemma, triStemm, stemm, lemma, clean
    # biPhrases 5, 10 default
    # triPhrases 5, 10 default
    # toLower: True default false
    settingsDict = {'columnList': ['title', 'shortText', 'richText'],
                    'gensimPreProcess': 'triLemma',
                    'allowedTags': ['NOUN', 'VERB'],}
    
    corpusDict = {'noBelow': 20,
                  'noAbove':0.4,
                  'keepN':1000}
    
    extraDict = {'perWordTopic': False,
                 'removeWords': ['bundes'], #Bug: Wörter mit Umlauten wie 'für' lassen sich nicht filtern.
                  'alpha': "symmetric",
                  'passes': 60}
    
    #   Data-Preperation:
    
    ##  Vervollständig das SettingsDict mit default Parametern
    settingsDict = dph.completeSettingsDict(settingsDict)
    
    filePath = dph.getDataPath(os.path.join('autoCreation', dph.getName(settingsDict)))
    modelPath = dph.getModelPath(os.path.join(modelType, str(uuid.uuid1())))
    
    ##  Erstellt die Pfade für die Daten und dem Model. Der Dateiname für die Daten wird aus dem SettingsDict zusammengesetzt.
    df = dph.dataCleaningPipeline('press42.json', settingsDict)
    df = dph.removingWords(df, settingsDict['gensimPreProcess'], extraDict['removeWords'])
    
    ##  Nachträgliches kleinschreiben der Wörter.
    if True:
        df = dph.toLower(df, settingsDict['gensimPreProcess'])
        extraDict['toLower'] = True
        
    ##  Erstellt das Bag of Words Model und ein Dictionary. // Hier wird das für auch entfernt.
    bowCorpus, dictionary = dph.getCorpus(df[settingsDict['gensimPreProcess']], corpusDict['noBelow'], corpusDict['noAbove'], corpusDict['keepN'])

    ##  Anwendung des tf-idfs.
    if True:
        tfidf = gensim.models.TfidfModel(bowCorpus)
        bowCorpus = tfidf[bowCorpus]
        extraDict['tfidf'] = True
    
    #   Ende der Data-Prepartion
    #   Training des Models
    
    ##  Initzialisiert die Trainingsphase für den Algorithmus. Die Parameter sollten aus dem extraDict kommen, falls nicht default.
    ldaModel = gensim.models.LdaModel(bowCorpus, num_topics=numberTopics, random_state=100, id2word=dictionary, alpha=extraDict['alpha'], passes=extraDict['passes'], iterations=100, per_word_topics=extraDict['perWordTopic'])
    
    #   Ende des Trainings
    #   Validierung
    
    ##  Berechnung der Perplexity
    perplexity = ldaModel.log_perplexity(bowCorpus)

    ## Erstellt den Coherence-Score u_mass
    cm = CoherenceModel(model=ldaModel, corpus=bowCorpus, coherence='u_mass')
    coherence = cm.get_coherence()
    
    ## Erstellt eine Liste und ein String mit den erstellten Topics
    topicString = ""
    topicList = []
    for idx, topic in ldaModel.print_topics(-1):
        topicList.append(topic)
        topicString += '\nTopic: {} \nWords: {}'.format(idx, topic) 
    
    ## Speichert die Ergebnisse und den Modeltype in ein Dictionary
    resultDict = {'modelType': modelType,
                 'topicN': numberTopics,
                 'perplexity':perplexity,
                 'coherence': coherence,
                 'topicList':topicList}

    ## Berechnung und Ausgabe des CoherenceCV scores c_v - zeitintensiv
    if True:
        texts = [[dictionary[word_id] for word_id, freq in doc] for doc in bowCorpus]
        cv = CoherenceModel(model=ldaModel, texts=texts, dictionary=dictionary, corpus=bowCorpus, coherence='c_v')
        coherenceCV = cv.get_coherence()
        resultDict['coherenceCV'] = coherenceCV
        print('\nCoherenceCV Score:', coherenceCV)
    
    ## Ausgabe der Resultate
    print('\nCoherence Score:', coherence)
    print('\nPerplexity:', perplexity)
    print(topicString)
    
    #   Ende der Validierung
    #   Speicherung der Daten
    
    ##  Speichert den File und Model path
    pathDict = {'modelPath': modelPath,
               'dataPath': filePath}
    
    ##  Lädt das resultRecordDict und hängt daran die Ergebnisse und Parameter dieses Models an
    resultRecordDict = dph.getResultRecordDict()
    resultRecordDict = dph.appendResultRecordDict([settingsDict, corpusDict, pathDict, resultDict, extraDict], resultRecordDict)
    
    ##  Speichert das Model und das resultRecordDict
    if True:
        dph.saveResultRecordDict(resultRecordDict)
        dph.saveModelDirect(ldaModel, modelPath)
    
    ##  Zeigt den Plot
    if True:
        LDAvisPrepared = pyLDAvis.gensim.prepare(ldaModel, bowCorpus, dictionary)
        pyLDAvis.show(LDAvisPrepared)
