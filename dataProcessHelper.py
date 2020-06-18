# -*- coding: utf-8 -*-
"""
Created on Fri May  8 02:03:30 2020

@author: mahes

DataProcessHelper dient mit seinen Funktionen als Hilfsmittel für die Data-Preperation-Schritte, 
zum Speichern und Laden von trainierten Modellen und für die Ausgabe von Topics für bestimmte Modele. 

"""
import string
import os
import pandas as pd
import numpy as np
import json

#from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
#from nltk.corpus import stopwords
import de_core_news_sm
nlp = de_core_news_sm.load()
import pickle
import locale
locale.setlocale(locale.LC_ALL, 'de_DE')
import gensim
from nltk.stem.cistem import Cistem

gensimPreprocessList = ['biLemma', 'triLemma', 'biStemm', 'triStemm']



def saveResultRecordDict(dict_):
    """
    Speichert das ResultRecord-Dictionary, welche die Ergebnisse und Parameter der trainierten Modelle enthält sowie
    den Pfad zu den Daten und dem gespeicherten Model.


    Parameters
    ----------
    dict_ : TYPE
        Dictionary mit den Ergebnissen und Parameter der trainierten Modelle sowie
        den Pfad zu den Daten und dem gespeicherten Model.

    Returns
    -------
    None.

    """
    filePath = os.path.join('result', 'resultRecordDict.json')
    with open(filePath, 'w', encoding='utf-8') as outfile:
        json.dump(dict_, outfile)

def getResultRecordDict():
    """
        Gibt die Datei aus, welche die Ergebnisse und Parameter der trainierten Modelle enthält sowie
        den Pfad zu den Daten und dem gespeicherten Model.

    Returns
    -------
    Dictionary
        Gibt die Datei aus, wenn sie existiert, welche die Ergebnisse und Parameter der trainierten Modelle enthält sowie
        den Pfad zu den Daten und dem gespeicherten Model. Ansonsten wird ein leeres Dictionary zurückgegeben.

    """
    filePath = os.path.join('result', 'resultRecordDict.json')
    if os.path.exists(filePath):
        with open(filePath) as json_file:
            dict_ = json.load(json_file)
        return dict_
    else:
        return dict()
    
def appendResultRecordDict(dictList, dict_):
    """
    Fusioniert eine Liste von Dictionaries zu einem eigenen Dictionary und hängt dieses als
    Record an ein anderes Dictionary. Es ist eine robusstere Alternative zu de der createResultRecord Funktion.

    Parameters
    ----------
    dictList : List<Dictionary>
        Eine Liste von Dictionaries mit den Ergebnissen und Parametern.
    dict_ : Dictionary
        Das Result-Dictionary.

    Returns
    -------
    dict_ : Dictionary
        Fusioniertes Dictionary.

    """
    count = len(dict_)
    recordDict = {}
    for tempDict in dictList:
        recordDict = {**recordDict, **tempDict}
    dict_[count] = recordDict
    return dict_
        
def dataCleaningPipeline(fileName, settingsDict, save=True):
    """
    Lädt die Datenbasis als Dataframe. Falls die spezifische Data-Preparation noch nicht
    durchgeführt oder gespeichert wurde, wird sie erstellt ansonsten wird sie geladen.
    Die Überprüfung finden anhand eines Namens für die vorverarbeitete Datenbasis statt, 
    der aus dem  SettingsDict generiert wird.

    Parameters
    ----------
    fileName : String
        Database name.
    settingsDict : Dictionary
        Einstellung für die DataPreparation.
    save : bool, optional
        Soll die Datenbasis nach der Bearbeitung gespeichert werden. The default is True.

    Returns
    -------
    df : Dataframe
        Resultat der Datenbearbeitung.

    """
    path = getDataPath(fileName)
    df = pd.read_json(path)

    filename = getName(settingsDict)
    filepath = getDataPath(os.path.join('autoCreation', filename))
    if os.path.exists(filepath):
        df = pd.read_json(filepath)
    else:
        toLower = bool(settingsDict['toLower']) if 'toLower' in settingsDict.keys() else False
        df['combText'] = df.apply(lambda row: combineColumnStrings(df, row, settingsDict['columnList'], toLower), axis=1)
        df = gensimPreProcess(df, settingsDict['gensimPreProcess'], settingsDict)
        df.drop('combText', inplace = True, axis=1)
        if (save == True):
            saveAsJson(df, os.path.join('autoCreation', filename))
    return df

#https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
def gensimPreProcess(df, switch, settingsDict):
    """
    Ruft die Methoden für die Data-Preparation auf wie das entfernen von Stoppwörtern und Zeichen und
    erstellt falls angegeben N-Grams und die 'Grundform' (Lemma, Stemming).

    Parameters
    ----------
    df : Dataframe
        Die unveränderte Datenbasis.
    switch : String
        Der Parameter, welche die Methode vorgibt. 
        clean, lemma, stemm, biLemma, biStemm, triLamme, triStemm 
    settingsDict : Dictionary
        Enthält die Parameter für die Erstellung der N-Grams.

    Returns
    -------
    df : Dataframe
        Das Resultat der ausgewählten Methode.

    """
    print('Log: create:', switch)
    
    allowedTags = settingsDict['allowedTags']
    biCount, biThres = getBiPhrases(settingsDict)
    trCount, trTrhes = getTriPhrases(settingsDict)
    
    if (switch in gensimPreprocessList):
        df['combClean'] = df.apply(lambda row: removeStopPunctSpace(df['combText'].iloc[row.name]), axis=1)
        
        bigram = gensim.models.Phrases(df['combText'], min_count=biCount, threshold=biThres) #8
        bigramMod = gensim.models.phrases.Phraser(bigram)
        dataWordsBigram = createBigrams(df['combClean'], bigramMod)
        #Evtl. erst die Lemme und Stemm bilden und dann Bigrams und Trigrams erzeugen.
        if (switch == 'biLemma'):
            df[switch] = lemmatization(dataWordsBigram, allowedTags)
        elif (switch == 'biStemm'):
            df[switch] = stemming(dataWordsBigram, allowedTags)
        elif(switch in ['triLemma', 'triStemm']):
            trigram = gensim.models.Phrases(bigram[df['combText']], min_count=trCount, threshold=trTrhes)
            trigramMod = gensim.models.phrases.Phraser(trigram)
            dataWordsTrigram = createTrigrams(df['combClean'], bigramMod, trigramMod)
            if (switch == 'triLemma'):
                df[switch] = lemmatization(dataWordsTrigram, allowedTags)
            else:
                df[switch] = stemming(dataWordsTrigram, allowedTags)
        return df
    elif (switch == 'stemm'):
        df[switch] = df.apply(lambda row: createStopPunctStemm(df['combText'].iloc[row.name], allowedTags), axis=1)
        return df
    elif (switch == "lemma"):
        df[switch] = df.apply(lambda row: createStopPunctLemma(df['combText'].iloc[row.name], allowedTags), axis=1)
        return df
    elif (switch == "clean"):
        df[switch] = df.apply(lambda row: removeStopPunctSpace(df['combText'].iloc[row.name], allowedTags), axis=1)
        return df
    else:
        df[switch] = df['combText']
        return df

def removingWordsSpecific(df, column, filterList = []):
    """
    Löscht die in der filterList enthaltende Wörter aus der Textspalte.
    Bug: Wörter mit Umlauten wie 'für' lassen sich nicht hier filtern.
    
    Parameters
    ----------
    df : Dataframe
        Zu bearbeitendes Dataframe.
    column : String
        Die Spalte, welche bearbeitet werden soll.
    filterList : List<string>, optional
        Eine Liste von Wörtern, welche gelöscht werden sollen. The default is [].

    Returns
    -------
    df : Dataframe
        Resultat der Löschung.

    """
    if filterList:
        for item in filterList:
            df[column].apply(lambda x: x.remove(item) if item in x else x)
    return df


def removingWords(df, column, filterList = []):
    """
    Überprüft ob ein Wort die in der filterList einthaltenden Zeichen beinhaltet, wenn ja
    dann wird das gesamte Wort gelöscht.
    Bug: Wörter mit Umlauten wie 'für' lassen sich nicht hier filtern.

    Parameters
    ----------
    df : Dataframe
        Zu bearbeitendes Dataframe.
    column : String
        Die Spalte, welche bearbeitet werden soll.
    filterList : List<string>, optional
        Eine Liste von Zeichen, die in den Wörtern enthalten sind, welche gelöscht werden sollen. The default is [].

    Returns
    -------
    df : Dataframe
        Resultat der Löschung.

    """
    if filterList:
        for item in filterList:
            for list_ in df[column]:
                for listElement in list_:
                    if item.lower() in listElement.lower():
                        list_.remove(listElement)
    return df

def toLower(df, column, toLower = True):
    """
    Schreibt alle Wörter klein.

    Parameters
    ----------
    df : Dataframe
        Zu bearbeitendes Dataframe.
    column : String
        Die Spalte, welche bearbeitet werden soll.
    toLower : bool, optional
        Ob die Methode angewandt werden soll. The default is True.

    Returns
    -------
    df : Dataframe
        Resultat der Kleinschreibung.

    """
    df[column] = df[column].apply(lambda row: [x.lower() for x in row]) if toLower == True else df[column]
    return df
        
def completeSettingsDict(settingsDict):
    """
    Vervollständigt das SettingsDict um die Parameter für die Bildung von N-Grams mit den Default values.
    Dies ist nötig damit diese nachher im ResultDict vorhanden sind.

    Parameters
    ----------
    settingsDict : Dictionary
        Settings Dictionary.

    Returns
    -------
    settingsDict : Dictionary
        Vervollständigtes Dictionary.

    """
    if settingsDict.get('gensimPreProcess') in gensimPreprocessList:
        if not 'biPhrases' in settingsDict.keys():
            settingsDict['biPhrases'] = [5, 10.0]
        if not 'trPhrases' in settingsDict.keys():
            settingsDict['trPhrases'] = [5, 10.0]
    return settingsDict

def getBiPhrases(settingsDict):
    """
    Gibt die Parameter für die Bildung von Bi-Grams zurück weil diese in einer Liste 
    im SettingsDict vorhanden sind.

    Parameters
    ----------
    settingsDict : Dictionary
        Settings Dictionary.

    Returns
    -------
    Int
        Min_count.
    Float
        threshold.

    """
    if ('biPhrases' in settingsDict.keys()):
        return settingsDict['biPhrases'][0], settingsDict['biPhrases'][1]
    else:
        return 5, 10.0
    
def getTriPhrases(settingsDict):
    """
    Gibt die Parameter für die Bildung von Tri-Grams zurück weil diese in einer Liste 
    im SettingsDict vorhanden sind.

    Parameters
    ----------
    settingsDict : Dictionary
        Settings Dictionary.

    Returns
    -------
    Int
        Min_count.
    Float
        threshold.

    """
    if ('trPhrases' in settingsDict.keys()):
        return settingsDict['trPhrases'][0], settingsDict['trPhrases'][1]
    else:
        return 5, 10.0

def getName(settingsDict):
    """
    Sortiert das Dictionary nach den Namen und erstellt einen String aus der Kombination 
    key- sortierte values für jeden Eintrag im Dictionary. Dieser String wird zum
    Speichern der vorverarbeiteten Daten verwendet. Dieser String kann dann wieder mit den selben
    Einträgen im SettingsDict generiert und ermöglicht das Laden der schon gespeicherten Daten.

    Parameters
    ----------
    settingsDict : Dictionary
        Settings dictionary.

    Returns
    -------
    name : string
        Der generierte Name.

    """
    name = ""
    for key in sorted(settingsDict):
        name += " "
        if (type(settingsDict[key]) == list):
            name += key[0:5] + '-'
            if key in ['biPhrases', 'trPhrases']:
                for element in settingsDict[key]:
                    name += str(element)
            else:
                for element in sorted(settingsDict[key]):
                    name += element[0:5]
        else:
            if (type(settingsDict[key]) == bool):
                name += key[0:5] + '-' + settingsDict[key]
            else:
                name += key[0:5] + '-' + settingsDict[key][0:5]
    name = name.strip().lower() 
    name += '.json'
    return name

def isStopPunctSpace(token):
    """
    Überprüft ob der Token ein Stoppwort, ein Satzzeichen, ein Leerzeichen oder eine Nummer ist.

    Parameters
    ----------
    token : spacy token
        Spacy token.

    Returns
    -------
    bool
        Gibt True zurück wenn Bedingung erfüllt.

    """
    return token.is_punct or token.is_space or token.is_stop or token.like_num

def combineColumnStrings(df, row, columnList, toLower=False):
    """
    Kombiniert die Dokumente angegebener Spalten innerhalb eines Datensatzes zu einem
    string. Handelt es sich um die Spalte 'richText' wird zusätzlich noch eine Filtermethode
    aufgerufen.

    Parameters
    ----------
    df : Dataframe
        Das Dataframe in dem auch der Datensatz ist.
    row : row
        Datensatz der kompiniert werden soll.
    columnList : List<string>
        Liste von Spaltennamen die beachtet werden sollen.
    toLower : bool, optional
        Gibt, an ob der Text kleingeschrieben werden soll. The default is False.

    Returns
    -------
    string
        Der zusammengesetzte string.

    """
    resultText = ""
    for column in columnList:
        if (column == 'richText'):
            richText = df[column].iloc[row.name]
            if not (isRichtextInFilterList(richText)):
                resultText += " " + str(removeImagePart(df[column].iloc[row.name]))
        else:
            resultText += " " + str(df[column].iloc[row.name])
    if toLower:
        return resultText.lstrip().lower();
    return resultText.lstrip()

def combineString(df, row):
    """
    Kombiniert aus einem Datensatz den ShortText und den bereinigten Richtext miteinander

    Parameters
    ----------
    df : Dataframe
        Dataframe des Datensatzes.
    row : row
        Der Datensatz aus dem die Texte verbunden werden sollen.

    Returns
    -------
    string
        Der zusammengesetzte String.

    """
    return df["shortText"].iloc[row.name] + " " + removeImagePart(df['richText'].iloc[row.name])

def isRichtextInFilterList(string_, filterList=['Zur externen Meldung', 'Zur externen Pressemitteilung']):
    """
    Gibt true aus wenn die Zeichenketten in der Filterlist im string vorhanden ist

    Parameters
    ----------
    string_ : string
        String der überprüft werden soll.
    filterList : list<string>, optional
        Liste von String zur Überprüfung. The default is ['Zur externen Meldung', 'Zur externen Pressemitteilung'].

    Returns
    -------
    bool
        Gibt true aus wenn die Zeichenketten in der Filterlist im string vorhanden ist.

    """
    return True if string_.strip() in filterList else False

#Prüft ob ein String einen Wert hat.
def isEmpty(string_):
    """
    Gibt true aus wenn der string leer ist

    Parameters
    ----------
    string_ : string
        String der überpüft werden soll.

    Returns
    -------
    bool
        Gibt true aus wenn der string leer ist.

    """
    return True if string_.str.len() == 0 else False


def removeImagePart(richText):
    """
    Löscht die Artefakte, die am Ende bestimmter Nachrichten durch das Foto entstanden sind.

    Parameters
    ----------
    richText : string
        Der string der bearbeitet werden soll.

    Returns
    -------
    string
        Der bearbeitete Text, falls "Foto" im Text ist ansonsten der unveränderte Text.

    """
    if ('Foto: ' in richText):
        list_ = richText.split('Foto: ')
        return list_[0] if ((len(list_[1]) < 70) & ('Zur externen Meldung' in list_[1])) else richText
    else: 
        return richText


def saveAsJson(df, filename):
    """
    Speicherte einen Dataframe als json

    Parameters
    ----------
    df : dataframe
        Das zu speicherende Dataframe.
    filename : string
        Name der Datei.

    Returns
    -------
    None.

    """
    with open(getDataPath(filename), 'w', encoding='utf-8') as file:
        df.to_json(file, force_ascii=False)

def saveModel(model, modelname):
    """
    Speichert das Model in model/

    Parameters
    ----------
    model : model
        Trainierte Model.
    modelname : string
        Name des Models.

    Returns
    -------
    None.

    """
    with open(getModelPath(modelname), 'wb') as f:
        pickle.dump(model, f)

def saveModelDirect(model, path):
    """
    Speichert das Model

    Parameters
    ----------
    model : Model
        Trainiertes Model.
    path : string
        Pfad mit Modelnamen.

    Returns
    -------
    None.

    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)
       
def loadModel(modelname):
    """
    Lädt ein Model im Model Ordner

    Parameters
    ----------
    modelname : string
        Modelname.

    Returns
    -------
    Model
        Trainiertes Model.

    """
    with open(getModelPath(modelname), 'rb') as f:
        return pickle.load(f)

def loadModelDirect(path):
    """
    Lädt ein Model mit Pfadangabe relativ

    Parameters
    ----------
    path : string
        Pfad + name.

    Returns
    -------
    model
        Trainiertes Model.

    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def createStopPunctLemma(text, allowedTags=None):
    """
    Diese Funktion wird aufgerufen, wenn keine Bi oder Trigrams verwendet wird.
    Erstellt Tokens und löscht nicht notwendige Satzzeichen und andere Zeichen.
    Erstellt Lemma und filtert die Wortart falls angegeben.

    Parameters
    ----------
    text : string
        Text der bearbeitet werden soll.
    allowedTags : list<string>, optional
        Wortart die beibehaltet werden soll. The default is None.

    Returns
    -------
    tokens : list<token>
        Eine Liste von gefilterten und geänderten Tokens.

    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not isStopPunctSpace(token):   
            if ((allowedTags == 'all') or (allowedTags is None)):
                tokens.append(token.lemma_)
            else:
                if token.pos_ in allowedTags:
                    tokens.append(token.lemma_)
    return tokens

def createStopPunctStemm(text, allowedTags=None):
    """
    Erstellt Tokens und löscht nicht notwendige Satzzeichen und andere Zeichen.
    Stemming mit cistem und filtert die Wortart falls angegeben.

    Parameters
    ----------
    text : string
        Text der bearbeitet werden soll.
    allowedTags : list<string>, optional
        Wortart die beibehaltet werden soll. The default is None.

    Returns
    -------
    tokens : list<token>
        Eine Liste von gefilterten und geänderten Tokens.

    """
    doc = nlp(text)
    stemmer = Cistem()
    tokens = []
    for token in doc:
        if not isStopPunctSpace(token):    
            if ((allowedTags == 'all') or (allowedTags is None)):
                    tokens.append(stemmer.segment(token.text)[0])
            else:
                if token.pos_ in allowedTags:
                    tokens.append(stemmer.segment(token.text)[0])
    return tokens


def getDataPath(filename):
    """
    Einfache Funktion für Pfad des Datenordner.

    Parameters
    ----------
    filename : string
        Dateiname.

    Returns
    -------
    string
        Der Pfad zur Datei.

    """
    return os.path.join('data', filename)

#Einfache Funktion für den Pfad der Modelle
def getModelPath(modelname):
    """
    Einfache Funktion für den Pfad des Modelordners

    Parameters
    ----------
    modelname : string
        Modelname.

    Returns
    -------
    string
        Der Pfad zum Model.

    """
    return os.path.join('model', modelname)

#Ähnliche Funnktion wie 'createStopPunctLemma' bloß ohne Lemma.
def removeStopPunctSpace(texts, allowedTags=None):
    """
    Erstellt Tokens und löscht nicht notwendige Satzzeichen und andere Zeichen.
    Filtert die Wortart falls angegeben.   

    Parameters
    ----------
    texts : string
        Text der bearbeitet werden soll.
    allowedTags : list<string>, optional
        Wortart die beibehaltet werden soll. The default is None.

    Returns
    -------
    tokens : list<token>
        Eine Liste von gefilterten und geänderten Tokens.

    """
    doc = nlp(texts)
    tokens = []
    for token in doc:
        if not isStopPunctSpace(token):
            if ((allowedTags == 'all') or (allowedTags is None)):
                    tokens.append(token.text)
            else:
                if token.pos_ in allowedTags:
                    tokens.append(token.text)
    return tokens

#Erstellt Bigrams
def createBigrams(texts, bigramMod):
    """
    Erstellt bigrams mithilfe des Gensim bigram models

    Parameters
    ----------
    texts : string
        Text der bearbeitet werden soll.
    bigramMod : Gensim Bigram Model
        Das initialisierte Bigram model.

    Returns
    -------
    bigramList : list<string>
        Erstelle Bigrams.

    """
    bigramList = []
    for doc in texts:
        tempResult = bigramMod[doc]
        bigramList.append(tempResult)
    return bigramList

#Erstellt aus den Bigrams - Trigrams.
def createTrigrams(texts, bigramMod, trigramMod):
    """
    Erstellt trigrams mithilfe des Gensim bigram und trigram models

    Parameters
    ----------
    texts : string
        Text der bearbeitet werden soll.
    bigramMod : Gensim Bigram Model
        Das initialisierte Bigram Model.
    trigramMod : Gensim Trigram Model
        Das initialisierte Bigram Model.

    Returns
    -------
    trigramList : list<string>
        Erstellte Trigrams.

    """
    trigramList = []
    for doc in texts:
        tempResult = trigramMod[bigramMod[doc]]
        trigramList.append(tempResult)
    return trigramList


def lemmatization(texts, allowedTags = None):
    """
    Erstellt die Lemma und filtert nach Wortart

    Parameters
    ----------
    texts : string
        Text der bearbeitet werden soll.
    allowedTags : list<string>, optional
        Wortart die beibehaltet werden soll. The default is None.

    Returns
    -------
    tokens : list<token>
        Eine Liste von gefilterten und geänderten Tokens.

    """
    tokens = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        if ((allowedTags == 'all') or (allowedTags is None)):
            tokens.append([token.lemma_ for token in doc])
        else:
            tokens.append([token.lemma_ for token in doc if token.pos_ in allowedTags])
    return tokens

def stemming(texts, allowedTags = None):
    """
    Stemming und filtert nach Wortart

    Parameters
    ----------
    texts : string
        Text der bearbeitet werden soll.
    allowedTags : list<string>, optional
        Wortart die beibehaltet werden soll. The default is None.

    Returns
    -------
    tokens : list<token>
        Eine Liste von gefilterten und geänderten Tokens.

    """
    tokens = []
    stemmer = Cistem()
    for sent in texts:
        doc = nlp(" ".join(sent))
        if ((allowedTags == 'all') or (allowedTags is None)):
            tokens.append([stemmer.segment(token.text)[0] for token in doc])
        else:
            tokens.append([stemmer.segment(token.text)[0] for token in doc if token.pos_ in allowedTags])
    return tokens    

#Gibt die für Gensim Modelle den notwendigen Output zurück.
def getCorpus(data, noBelow=5, noAbove=0.5, keepN=100000, keep_tokens=None):
    """
    Fitler extrem Werte und erstellt ein Dictionary sowie ein BagOfWord Modell

    Parameters
    ----------
    data : Series/Matrix 1d
        Die Daten welche verarbeitet werden.
    noBelow : int, optional
        Filtert alle Wörter die nicht in mindestens x Dokumenten sind. The default is 5.
    noAbove : float, optional
        Filtert Wörter die in x (0-1) der Dokumente enthalten sind.. The default is 0.5.
    keepN : int, optional
        Behalte x Wörter. The default is 100000.
    keep_tokens : list<string>, optional
        Behalte spezielle Wörter, die sonst rausgefiltert werden. The default is None.

    Returns
    -------
    bowCorpus : TYPE
        Bag of Words Model.
    dictionary : Gensim Dictionary
        Erstelltes Dictionary.

    """
    dictionary = gensim.corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=noBelow, no_above=noAbove, keep_n=keepN, keep_tokens=None)
    dictionary.compactify()
    bowCorpus = [dictionary.doc2bow(doc) for doc in data]
    return bowCorpus, dictionary


def getNmfLdaTopic(model, vectorizer, numberTopics, nTopWords):
    """
    Ermittelt die Topics für die Modelle NmF und LDA.

    Parameters
    ----------
    model : model
        Scikit Learn lda oder nmf model.
    vectorizer : vectorizer-model
        Zum Beispiel Tfi-idf oder Countvectorizer.
    numberTopics : int
        Anzahl der Topics.
    nTopWords : int
        Anzahl der Wörter in den Topics die angezeigt werden sollen.

    Returns
    -------
    Dataframe
        Result dataframe mit dem Topics.

    """
    term = vectorizer.get_feature_names()
    wordDict = {};
    for i in range(numberTopics):
        wordsIds = model.components_[i].argsort()[:-nTopWords - 1:-1]
        words = [term[key] for key in wordsIds]
        wordDict['Topic # ' + '{:02d}'.format(i+1)] = words;
    return pd.DataFrame(wordDict);


def getNmfLdaTopicsTermsWeights(weights, featureNames):
    """
    Erstellt die Topic-Gewichtungen für die Tokens für lda und nmf

    Parameters
    ----------
    weights : array<float>
        Output von model.components_
    featureNames : list
        output.get_feature_names()

    Returns
    -------
    topics : list<array>
        Eine Topic list mit den gewichteten Wörtern für die funktion printTopicUdf.

    """
    featureNames = np.array(featureNames)
    sortedIndices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sortedWeights = np.array([list(wt[index]) for wt, index in zip(weights, sortedIndices)])
    sortedTerms = np.array([list(featureNames[row]) for row in sortedIndices])

    topics = [np.vstack((terms.T, termWeights.T)).T for terms, termWeights in zip(sortedTerms, sortedWeights)]

    return topics

def printTopicsUdf(topics, numberTopics=1, weightThreshold=0.0001, displayWeights=False, numTerms=None):
    """
    Gibt die Topics aus mit ihren Token-Gewichtungen

    Parameters
    ----------
    topics : list<array>
        Ergebnis aus der Funktion getNmfLdaTopicsTermsWeights.
    numberTopics : int, optional
        Anzahl der Topics. The default is 1.
    weightThreshold : float, optional
        Schwelle für die Gewichtung. The default is 0.0001.
    displayWeights : bool, optional
        Ob die Gewichtung ausgegeben werden soll. The default is False.
    numTerms : int, optional
        Anzahl der auszugebenen Tokens. The default is None.

    Returns
    -------
    None.

    """
    for index in range(numberTopics):
        topic = topics[index]
        topic = [(term, float(wt)) for term, wt in topic]
        topic = [(word, round(wt,2)) for word, wt in topic if abs(wt) >= weightThreshold]
        if displayWeights:
            print('Topic #'+str(index+1)+' mit Gewichtung')
            print(topic[:numTerms]) if numTerms else topic
        else:
            print('Topic #'+str(index+1)+' ohne Gewichtung')
            tw = [term for term, wt in topic]
            print(tw[:numTerms]) if numTerms else tw
            
def getTopicsUdf(topics, numberTopics=1, weightThreshold=0.0001, numTerms=None):
    """
    Erzeugt eine Liste mit dem Topics und den Wortgewichtungen

    Parameters
    ----------
    topics : list<array>
        Ergebnis aus der Funktion getNmfLdaTopicsTermsWeights.
    numberTopics : int, optional
        Anzahl der Topics. The default is 1.
    weightThreshold : float, optional
        Schwelle für die Gewichtung. The default is 0.0001.
    numTerms : int, optional
        Anzahl der auszugebenen Tokens. The default is None.
        
    Returns
    -------
    topicTerms : list<array>
        Liste der Topics und der Wörter mit ihren Gewichtungen.

    """
    topicTerms = []
    for index in range(numberTopics):
        topic = topics[index]
        topic = [(term, float(wt)) for term, wt in topic]
        topic = [(word, round(wt,2)) for word, wt in topic if abs(wt) >= weightThreshold]
        topicTerms.append(topic[:numTerms] if numTerms else topic)
    return topicTerms



##################################
#### Ausgemusterte Funktionen ####
##################################



# 
# Die folgenden vier Funktionen sind zum Speichern der Data-Preparation-, Model-Parameter und des Resultates über ein Dataframe
# Aufgrund des Dataframes sind diese Funktionen instabil weil es Probleme gibt bei dem Speichern von Listen innerhalb des Dataframes.
# Die Probleme können auch dazu führen, dass die Datei korrupt wird.


# def dfEmpty(columns, dtypes, index=None):
#     assert len(columns)==len(dtypes)
#     df = pd.DataFrame(index=index)
#     for c,d in zip(columns, dtypes):
#         df[c] = pd.Series(dtype=d)
#     return df

# Ruft das Dataframe auf und überprüft, ob es existiert wenn nicht erstelle eins.
# def getResultDf():
#     file = os.path.join('result', 'resultDataframe.json')
#     if os.path.exists(file):
#         return pd.read_json(file)
#     else:
#         columnsPre =['columnList', 'gensimPreProcess', 'biPhrases', 'trPhrases', 
#                      'allowedTags', 'noBelow', 'noAbove', 'keepN', 'modelPath', 
#                      'dataPath', 'perplexity', 'coherence', 'topicList']
#         dftypePre = [object, str, object, object, 
#                      object, np.int64, np.int64, np.int64, 
#                      str, str, float, float, object ]

#         df = dfEmpty(columnsPre, dftypePre)
#         with open(file, 'w', encoding='utf-8') as file:
#             df.to_json(file, force_ascii=False)
#         return df

# Speichert das Dataframe
# def saveResultDf(df):
#     file = os.path.join('result', 'resultDataframe.json')
#     with open(file, 'w', encoding='utf-8') as file:
#         df.to_json(file, force_ascii=False)


# Erstellt aus einer Liste von Dictionaries einen Ergebnisrecord
# Nachdem ersten speicheren des Dataframes werden alle Listen mit nur einen Wert als String gespeichert. Grund???
# Aus diesem Grund die etwas kompliziertere Abfrage, damit sichergestellt wird, dass auch eine Liste mit einen Element als Liste in die Zelle kommt.
# Sehr fehleranfällig
# '''
# def createResultRecord(dictList, df):
#     index = 0 if df.empty else df.index[-1] + 1
#     df = df.append(pd.Series(), ignore_index=True)
#     for tempDict in dictList:
#         for key in tempDict.keys():
#             if type(tempDict.get(key)) == list and not index == 0:
#                 if len(tempDict.get(key)) == 1:
#                     df.iat[index, df.columns.get_loc(key)] = [tempDict.get(key)]
#                 else:
#                     df.iat[index, df.columns.get_loc(key)] = tempDict.get(key)
#             else:
#                 df.iat[index, df.columns.get_loc(key)] = tempDict.get(key)
#     return df



# def removeStopPunctSpace(texts):
#     doc = nlp(texts)
#     tokens = []
#     for token in doc:
#         if not isStopPunctSpace(token):
#             tokens.append(token.text)
#     return tokens

# def createStopPunctLemma(text):
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if not isStopPunctSpace(token):    
#             tokens.append(token.lemma_)
#     return tokens
# 