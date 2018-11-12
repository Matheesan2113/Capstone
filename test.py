import re
import numpy as numpy
import pandas as pd
from pprint import pprint 

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#Spacy for lemmatization
import spacy

#Plotting tools
import pyLDAvis
import pyLDAvis.gensim #Important
import matplotlib.pyplot as plt 
#%matplotlib inline

#Enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#NLTK stop words
from nltk.corpus import stopwords
stop_words= stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use'])

#Import Dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
print(df.target_names.unique())
#Show first 5 of df
df.head()


#Convert to List
data = df.content.values.tolist()

#Remove Emails
data = [re.sub('\S*@\S*\s?', '',sent) for sent in data]
#Remove new line characters
data = [re.sub('\s+',' ',sent) for sent in data]
#Remove distracting single qquotes
data = [re.sub("\'", "", sent) for sent in data]
#Print data from email after cleaning up with REGEX
pprint(data[:1])


#Break down each sentence into a list of words through TOKENIZATION
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) #Removes Punctuations
data_words = list(sent_to_words(data))
#result from breaking down each sentence into a list of words through TOKENIZATION
print(data_words[:1])


#Build bigram & Trigram models
bigram = gensim.models.Phrases(data_words,min_count=5,threshold=100) #Higher threshold = fewer phrases
trigram = gensim.models.Phrases(bigram[data_words],threshold=100)
#Faster way to get sentence clubbed as birgram/trigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
#See trigram example
print("-----------------------------")
print(trigram_mod[bigram_mod[data_words[0]]])


#define functions for stopwords, bi/trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN','ADJ','VERB','ADV']):
    """https://spacy.io/api/annotation"""
    texts_out= []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#remove Stop words
data_words_nostops = remove_stopwords(data_words)
#form bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
#finitialize spacy 'en' model
nlp = spacy.load('en', disable=['parser','ner'])
#Do lemmatization keeping only nouns, adjectives, verbs and adverbs
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#Print data after removing stop words, form bigram and lemmatizing data
print("-------------------------------------")
print(data_lemmatized[:1])


#Create Dict
id2word = corpora.Dictionary(data_lemmatized)
#Create Corpus
texts = data_lemmatized
#Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
#View
print(corpus[:1])