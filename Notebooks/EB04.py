#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# spacy for lemmatization
import spacy
import json
import warnings
import networkx as nx

warnings.filterwarnings("ignore",category=DeprecationWarning)

import nltk
nltk.download('stopwords')
nltk.download('words')

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.corpus import words
eng_words = words.words('en')

from langdetect import detect
# to enforce consistent results, check github langdetect readme
from langdetect import DetectorFactory
DetectorFactory.seed = 0


import os 

import requests


# In[ ]:


count = 0
array = []
def getTextFromJson():
    global count
    for file in os.listdir('tweetFiles'):
        print(file)
        if count < 200:
            with open('tweetFiles/' + file, "r") as f:
                for line in f:
                    for key, value in json.loads(line).items():
                        if(key =="text"):
                            if(len(value) >=100):
                                # get english tweets
                                try:
                                    if (detect(value)=="en"):
                                        # print (detect(value))
                                        array.append(value)
                                except:
                                    pass
        count = count + 1

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

getTextFromJson()

print("FINI")


# In[ ]:


data = array

# Remove links
data = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# Remove hashtag
data = [re.sub("#", "", sent) for sent in data]

#remove non-ascii characters need to fix leaves blank lines
#data = [re.sub(r'[^\x00-\x7F]+','', sent) for sent in data]

def remove_non_english_words(tweets):
     return [[word for word in simple_preprocess(str(tweet)) if word in eng_words] for tweet in tweets]

def remove_empty_sent(tweets) :
    res = []
    for tweet in tweets:
        if len(tweet) != 0:
            res.append(tweet)
    return res

#data = remove_non_english_words(data)

data = remove_empty_sent(data)

print("PLS")


# In[ ]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


data_words = list(sent_to_words(data))

data_words_no_stp = remove_stopwords(data_words)


print(data_words_no_stp[:1])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[ ]:



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
# nlp = spacy.load('en', disable=['parser', 'ner'])

# # Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_no_stp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# In[ ]:


with open('preprocesseddata.txt', 'w') as f:
    for row in data_lemmatized:
        f.write("%s\n" % row)


# In[2]:


import json
import ast
data_lemmatized = []
with open('preprocesseddata.txt', "r") as f:
    for line in f:
        data_lemmatized.append(ast.literal_eval(line))

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
# type(data_lemmatized)
# print (data_lemmatized)
texts = data_lemmatized

print(data_lemmatized[:1])


# In[ ]:


print (len (data_lemmatized))


# In[3]:


# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Human readable format of corpus (term-frequency)
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# Build LDA model
lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=20, passes=10, workers=7)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_no_stp, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[4]:



# Visualize the topics
#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')


# In[6]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)

limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[10]:


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        print(row) 
        topics = []
        row = sorted(row, key=lambda x: (x[1]), reverse=True) 
        for j, (topic_num, prop_topic) in enumerate(row):
            if j in [0,1,2]: 
                topics.append(topic_num)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
#             if j in [0,1,2]: #top 3 dominant topics
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([topics, round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_lemmatized)


# In[11]:


# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
#TER
# Show
df_dominant_topic.head(10)


# In[ ]:


# Number of Documents for Each Topic
#topic_counts=0
#pprint(df_topic_sents_keywords['Dominant_Topic'].value_counts())
# for y in range (0,3):
#     pprint(df_topic_sents_keywords['Dominant_Topic'].value_counts())
#topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

#print (topic_counts)

# Percentage of Documents for Each Topic
#topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
#topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
#df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
#df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
#df_dominant_topics


# In[13]:


for index, row in df_dominant_topic.iterrows():
    print (row["Dominant_Topic"])


# In[33]:


model = gensim.models.Word2Vec(
        data_lemmatized,
        size=150,
        window=10,
        min_count=1,
        workers=7)
model.train(data_lemmatized, total_examples=len(data_lemmatized), epochs=10)
    
words = []
words2 = []
    


# In[ ]:


#Networkx
import datetime
G = nx.Graph()
# G.add_node(1)
# G.add_nodes_from([2, 3])

count = 0
print(datetime.datetime.now())

for index, row in df_dominant_topic.iterrows():
    G.add_node(count)
    G.node[count]["text"] = row["Text"]
    G.node[count]["topic"] = row["Dominant_Topic"]
    count += 1
  #  if count > 70: break;
print(datetime.datetime.now())
for i in range(count):
    for j in range(i+1,count):
        words = G.node[i]["text"]
        words2 = G.node[j]["text"]
        sum = 0
        avgSimilarity = 0
        for x in words:
            for y in words2:
                if (len(words) !=0 and len(words2) !=0):
                    sum += model.wv.similarity(w1=x, w2=y)
        if (len(words) !=0 and len(words2) !=0):     
            avg = sum /(len(words) * len(words2))
#             print(avg)
        
        for z in range(0,len(G.node[i]["topic"])):
            if G.node[i]["topic"][z] in G.node[j]["topic"]:
                G.add_edge(i, j, weight=avg)
print(datetime.datetime.now())
print(nx.number_of_edges(G))
print(nx.number_of_nodes(G))

# print(G.node[16]["text"])
# print(G.node[16]["topic"])
# print(G.get_edge_data(0,16))
# print(G.node[0]["text"])
# print(G.node[0]["topic"])

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.2]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.2]


nx.draw(G, with_labels=True, font_weight='bold', node_size=100, edgelist=elarge,
                       width=1, edge_color='b')
nx.draw(G, with_labels=True, font_weight='bold', node_size=100, edgelist=esmall,edge_color='g')
plt.show()
print(datetime.datetime.now())


# In[52]:


G = nx.Graph()

G.add_edge('a', 'b', weight=0.6)
G.add_edge('a', 'c', weight=0.2)
G.add_edge('c', 'd', weight=0.1)
G.add_edge('c', 'e', weight=0.7)
G.add_edge('c', 'f', weight=0.9)
G.add_edge('a', 'd', weight=0.3)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=6)
nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=6, alpha=0.5, edge_color='b', style='dashed')

# labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis('off')
plt.show()
        


# In[87]:


print(nx.clustering(G))


# In[88]:


G=nx.complete_graph(5)
nx.draw(G)
plt.show()


# In[ ]:




