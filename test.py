import os
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

# Import Dataset
basePath = os.path.dirname(os.path.abspath(__file__))
print(basePath)
df = pd.read_json(basePath+'/59.json')
print(df.text.unique())
df.head()


#Convert to List
data = df.text.values.tolist()

#Remove Emails
#data = [re.sub('\S*@\S*\s?', '',sent) for sent in data]
#Remove new line characters
#data = [re.sub('\s+',' ',sent) for sent in data]
#Remove distracting single qquotes
#data = [re.sub("\'", "", sent) for sent in data]
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

#[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,
num_topics=15, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
print("TEST: DONE LDA MODEL BUILDING")

# Print the Keyword in the 20 topics
pprint(lda_model.print_topics())
#doc_lda = lda_model[corpus]


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)
print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

# Visualize the topics
#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')



#malletPath
mallet_path ='/mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
#Show topics
pprint(ldamallet.show_topics(formatted=False))
# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)
