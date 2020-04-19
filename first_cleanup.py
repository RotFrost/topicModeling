# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:59:00 2020

@author: mahes
"""

import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

path = 'press42.json'

df = pd.read_json(path)

exampleText = df.iloc[0].richText

#Stop words + 

stopWords = set(stopwords.words('german'))
words = word_tokenize(exampleText)
words = [word.lower() for word in words if word.isalpha()]
print(words)
wordsFiltered = []

for word in words:
    if word not in stopWords:
        
        
        wordsFiltered.append(word)

print(wordsFiltered)

#lemme

import spacy
nlp = spacy.load('de_core_news_md')

mails=['Hallo. Ich spielte am frühen Morgen und ging dann zu einem Freund. Auf Wiedersehen', 'Guten Tag Ich mochte Bälle und will etwas kaufen. Tschüss']

mails_lemma = []

for mail in mails:
     doc = nlp(mail)
     result = ''
     for token in doc:
        result += token.lemma_
        result += ' ' 
     mails_lemma.append(result)