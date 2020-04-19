# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:59:00 2020

@author: mahes
"""

import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
import string
import spacy
import de_core_news_sm
nlp = de_core_news_sm.load()

def isStopPunctSpace(token):
    return token.is_punct or token.is_space or token.is_stop

def combineStrings(df, row):
    return df["shortText"].iloc[row.name] + " " + removeImagePart(df['richText'].iloc[row.name])

def removeImagePart(richTextString):
    return richTextString.split('Foto:')[0] if len(richTextString) < 100 else richTextString

richTextFilter = ['Zur externen Meldung', 'Zur externen Pressemitteilung']

path = 'press42.json'

df = pd.read_json(path)


exampleText = df.iloc[0].richText


#translate_table = dict((ord(char), None) for char in string.punctuation)   
#exampleText.translate(translate_table)

#df['train'] = df.apply(lambda row: df['shortText'].iloc[row.name] if df['richText'].iloc[row.name] in richTextFilter else combineStrings(df, row), axis=1)

#dfMin = df['richText'].loc[df['richText'].str.len() < 120].unique()

#dfStory = df.loc[df['richText'] == dfMin[2]]


#print(dfMin[2].split('Foto')[0].rstrip())

#print(exampleText)
#Stop words nltk
'''
stopWords = set(stopwords.words('german'))
words = word_tokenize(exampleText)
words = [word.lower() for word in words if word.isalpha()]
wordsFiltered = []

print(words)

for word in words:
    if word not in stopWords:
        wordsFiltered.append(word)

print(wordsFiltered)
'''
#Stop words spacy
# 
'''
doc = nlp(exampleText)
tokens = [token.text for token in doc if not token.is_stop]

print(tokens)

doc = nlp(exampleText)
tokens = []
for token in doc:
    if not isStopPunctSpace(token):
        tokens.append(token.lemma_)

print(tokens)

#lemme

import spacy
import de_core_news_sm
nlp = de_core_news_sm.load()


mails=['Hallo. Ich spielte am frühen Morgen und ging dann zu einem Freund. Auf Wiedersehen', 'Guten Tag Ich mochte Bälle und will etwas kaufen. Tschüss']

mails_lemma = []

for mail in mails:
     doc = nlp(mail)
     result = ''
     for token in doc:
        result += token.lemma_
        result += ' ' 
     mails_lemma.append(result)
     
print(mails_lemma)
'''