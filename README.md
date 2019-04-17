
# **Capstone EB04**

The following version of the code is for finding like minded user communities using regular lda analysis in order to find topics combined with clustering. This is the first of two different methods that were attempted. This method is without meta level topics.

## Imports


```python
import os 
import csv
import json
import datetime
import ast
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import models

# spacy for lemmatization
import spacy
import json
import warnings
import networkx as nx

warnings.filterwarnings("ignore",category=DeprecationWarning)

from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0
import numpy as np
import pandas as pd
from pprint import pprint
import pickle


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords')
nltk.download('words')

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.corpus import words
eng_words = words.words('en')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/ragulan550/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package words to /home/ragulan550/nltk_data...
    [nltk_data]   Package words is already up-to-date!


## Load Preprocessed User Tweet Data


```python
#loads csv from stored location
df = pd.read_csv('../csvfiles/tweetsOnUserOnConcepts.csv', lineterminator='\n', low_memory=False)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>TweetText</th>
      <th>ConceptText</th>
      <th>userid</th>
      <th>creationtimestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11123785801404416</td>
      <td>tweet RT</td>
      <td>Twitter RT (TV network)</td>
      <td>142685766</td>
      <td>2010-12-04 18:24:51 UTC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16801364319404032</td>
      <td>WHITE</td>
      <td>White American</td>
      <td>81450435</td>
      <td>2010-12-20 10:25:32 UTC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8977557407928321</td>
      <td>smiley face</td>
      <td>Smiley</td>
      <td>89099440</td>
      <td>2010-11-28 20:16:31 UTC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10017331400941568</td>
      <td>kami sama rin</td>
      <td>Kami Japanese honorifics Japanese yen</td>
      <td>142962699</td>
      <td>2010-12-01 17:08:12 UTC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18759561343143936</td>
      <td>king William the Conqueror England crowned</td>
      <td>Charles I of England William the Conqueror Kin...</td>
      <td>22619937</td>
      <td>2010-12-25 20:06:42 UTC</td>
    </tr>
  </tbody>
</table>
</div>




```python
#gets all concept text for each tweet and stores in list
tweetConcept = df.ConceptText.values.tolist()
userIds = df.userid.values.tolist()
```


```python
#for each users stores a list of their tweets stored word by word in a dictionary
dictConcept = {}
for i in range(len(userIds)):
    if userIds[i] not in dictConcept:
        dictConcept[userIds[i]] = []
    for word in str(tweetConcept[i]).split(" "):
        dictConcept[userIds[i]].append(word)
```

## LDA Analysis


```python
#list of all tweets for a user
data_final = list(dictConcept.values())

#setting up corpus for lda
id2word = corpora.Dictionary(data_final)
texts = data_final
corpus = [id2word.doc2bow(text) for text in texts]
```

### Run LDA
Multicore allows for multiple cores to be working on LDA simultaneously
- Check Number of workers
- Check Number of topics set<br>


```python
#uncomment line below to try lda with different values
topicNum = 47
#lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=topicNum, passes=10, workers=7)
#lda_model.save('../LdaSaves/topics47mar10p9/lda.model_mar10_t47')

#preloading an saved lda run to save time as lda takes long time to run

lda_model =  models.LdaModel.load('../LDASaves/topics47mar10p9/lda.model_mar10_t47')
pprint(lda_model.print_topics())
```

    [(16,
      '0.444*"Twitter" + 0.053*"TV" + 0.052*"series)" + 0.050*"This" + '
      '0.047*"language" + 0.047*"Week" + 0.045*"(ABC" + 0.037*"ATP" + '
      '0.037*"Celebrity" + 0.014*"of"'),
     (2,
      '0.219*"Blog" + 0.044*"(magazine)" + 0.038*"(service)" + 0.035*"Blogger" + '
      '0.023*"Don" + 0.019*"Tom" + 0.016*"Veja" + 0.014*"Club" + 0.014*"Cruise" + '
      '0.012*"Electronic"'),
     (45,
      '0.025*"of" + 0.017*"China" + 0.016*"United" + 0.014*"Bank" + 0.014*"States" '
      '+ 0.009*"Ireland" + 0.009*"The" + 0.009*"European" + 0.008*"Federal" + '
      '0.008*"and"'),
     (36,
      '0.063*"F.C." + 0.023*"football" + 0.019*"team" + 0.017*"Manchester" + '
      '0.015*"national" + 0.015*"Association" + 0.014*"Cup" + 0.013*"FIFA" + '
      '0.012*"League" + 0.012*"FC"'),
     (18,
      '0.023*"of" + 0.021*"de" + 0.013*"São" + 0.012*"Paulo" + 0.010*"European" + '
      '0.009*"Rio" + 0.009*"Marcus" + 0.009*"Pode" + 0.008*"da" + 0.008*"do"'),
     (46,
      '0.029*"BBC" + 0.023*"of" + 0.020*"The" + 0.018*"United" + '
      '0.017*"Photography" + 0.013*"States" + 0.010*"Africa" + 0.010*"the" + '
      '0.009*"San" + 0.009*"Francisco"'),
     (32,
      '0.027*"California" + 0.015*"of" + 0.013*"Florida" + 0.013*"States" + '
      '0.013*"United" + 0.011*"New" + 0.010*"San" + 0.008*"Angeles" + 0.008*"Los" '
      '+ 0.008*"County,"'),
     (15,
      '0.043*"Israel" + 0.020*"Gaza" + 0.020*"of" + 0.019*"people" + '
      '0.017*"Canada" + 0.014*"Palestinian" + 0.013*"The" + 0.012*"United" + '
      '0.011*"Egypt" + 0.010*"Palestine"'),
     (23,
      '0.016*"of" + 0.014*"The" + 0.012*"Intelligence" + 0.010*"the" + '
      '0.010*"Directorate" + 0.010*"(Israel)" + 0.009*"Military" + 0.009*"God" + '
      '0.008*"Love" + 0.008*"(philosophy)"'),
     (1,
      '0.025*"Physical" + 0.016*"The" + 0.014*"Health" + 0.013*"Vampire" + '
      '0.012*"fitness" + 0.012*"Weight" + 0.011*"Obesity" + 0.010*"exercise" + '
      '0.009*"Sexual" + 0.009*"loss"'),
     (19,
      '0.115*"New" + 0.066*"York" + 0.023*"Chicago" + 0.022*"City" + '
      '0.021*"Jersey" + 0.011*"of" + 0.011*"The" + 0.008*"Illinois" + '
      '0.008*"Brooklyn" + 0.007*"Blizzard"'),
     (41,
      '0.035*"Korea" + 0.025*"of" + 0.024*"United" + 0.021*"South" + 0.016*"North" '
      '+ 0.015*"China" + 0.013*"States" + 0.009*"WikiLeaks" + 0.009*"News" + '
      '0.008*"The"'),
     (13,
      '0.062*"Hawaii" + 0.023*"of" + 0.018*"Aloha" + 0.016*"Five-0" + 0.013*"List" '
      '+ 0.012*"sports" + 0.012*"Pearl" + 0.012*"figures" + 0.012*"attendance" + '
      '0.011*"Harbor"'),
     (3,
      '0.020*"Sudan" + 0.019*"Child" + 0.016*"violence" + 0.015*"Domestic" + '
      '0.014*"of" + 0.014*"Rape" + 0.014*"Human" + 0.012*"Violence" + '
      '0.012*"Virginia" + 0.011*"Woman"'),
     (44,
      '0.035*"of" + 0.025*"BBC" + 0.023*"The" + 0.022*"News" + 0.018*"Kingdom" + '
      '0.015*"London" + 0.015*"United" + 0.013*"(UK)" + 0.011*"the" + 0.010*"UK"'),
     (37,
      '0.040*"Indonesia" + 0.031*"Jakarta" + 0.027*"of" + 0.016*"people" + '
      '0.015*"Malaysia" + 0.015*"International" + 0.013*"Union" + '
      '0.012*"Telecommunication" + 0.011*"language" + 0.011*"Persian"'),
     (12,
      '0.078*"International" + 0.059*"Airport" + 0.038*"MCOT" + 0.035*"Name" + '
      '0.032*"Nonproprietary" + 0.023*"Service" + 0.020*"London" + 0.020*"Public" '
      '+ 0.018*"Thai" + 0.016*"Broadcasting"'),
     (22,
      '0.096*"Don\'t" + 0.047*"don\'t" + 0.047*"tell" + 0.047*"ask," + '
      '0.033*"LGBT" + 0.030*"Gay" + 0.025*"of" + 0.024*"Act" + 0.023*"Repeal" + '
      '0.023*"Tell"'),
     (35,
      '0.051*"card" + 0.041*"Belarus" + 0.031*"Credit" + 0.028*"Recycling" + '
      '0.023*"Visa" + 0.022*"Argentina" + 0.022*"Plastic" + 0.020*"Gift" + '
      '0.017*"Electron" + 0.016*"Russia"'),
     (7,
      '0.037*"Weather" + 0.033*"Snow" + 0.024*"Missouri" + 0.023*"Winter" + '
      '0.022*"County," + 0.022*"Carolina" + 0.021*"North" + 0.020*"Wind" + '
      '0.019*"Rain" + 0.017*"Zone"')]


#### Compute Perplexity and Coherence


```python
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_final, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
```

    
    Perplexity:  -9.179285768777685
    
    Coherence Score:  0.50180613135149


#### Finding Topic Distribution


```python
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_final):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_final)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-8-36a28681a47a> in <module>
         22 
         23 
    ---> 24 df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_final)
         25 
         26 # Format


    <ipython-input-8-36a28681a47a> in format_topics_sentences(ldamodel, corpus, texts)
          9         for j, (topic_num, prop_topic) in enumerate(row):
         10             if j == 0:  # => dominant topic
    ---> 11                 wp = ldamodel.show_topic(topic_num)
         12                 topic_keywords = ", ".join([word for word, prop in wp])
         13                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)


    ~/.local/lib/python3.5/site-packages/gensim/models/ldamodel.py in show_topic(self, topicid, topn)
       1190 
       1191         """
    -> 1192         return [(self.id2word[id], value) for id, value in self.get_topic_terms(topicid, topn)]
       1193 
       1194     def get_topics(self):


    ~/.local/lib/python3.5/site-packages/gensim/models/ldamodel.py in get_topic_terms(self, topicid, topn)
       1222 
       1223         """
    -> 1224         topic = self.get_topics()[topicid]
       1225         topic = topic / topic.sum()  # normalize to probability distribution
       1226         bestn = matutils.argsort(topic, topn, reverse=True)


    ~/.local/lib/python3.5/site-packages/gensim/models/ldamodel.py in get_topics(self)
       1202 
       1203         """
    -> 1204         topics = self.state.get_lambda()
       1205         return topics / topics.sum(axis=1)[:, None]
       1206 


    ~/.local/lib/python3.5/site-packages/gensim/models/ldamodel.py in get_lambda(self)
        267 
        268         """
    --> 269         return self.eta + self.sstats
        270 
        271     def get_Elogbeta(self):


    KeyboardInterrupt: 



```python
# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics[0:47]
```

#### Creating User Vectors of length K where K is number of topics


```python
UserVectors = []

#for each users shows percent contribution for that topic
print(lda_model[corpus][1][1])

for row in lda_model[corpus]:
    temp = [0]*topicNum
    for val in row:
        #val is a tuple in form (topicNum, percentContributionOfTopicToUser)
        temp[val[0]] = val[1]
    UserVectors.append(temp)
    
print("Shows a sample userVector")    
print(UserVectors[1])
```

    (1, 0.010639478)
    Shows a sample userVector
    [0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.5105822, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517, 0.010639517]


## Load Preprocessed Gold Standard News Articles


```python
dfGoldStandard = pd.read_csv('../csvfiles/GoldStandard.csv',  lineterminator='\n', low_memory=False)
dfGoldStandard.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>tweetid</th>
      <th>userid</th>
      <th>creationtimestamp</th>
      <th>NewsId</th>
      <th>NewsText</th>
      <th>NewsConceptText</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>http://on.cnn.com/9BMsbh</td>
      <td>5813001621872640</td>
      <td>32814009</td>
      <td>2010-11-20 02:41:42 UTC</td>
      <td>50637</td>
      <td>pharmaceutical companies Big Pharma OH MY GAWD...</td>
      <td>Pharmaceutical industry Pharmaceutical industr...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://money.cnn.com/2010/12/23/pf/rich_wealth...</td>
      <td>18123124054695937</td>
      <td>18097177</td>
      <td>2010-12-24 01:57:44 UTC</td>
      <td>76310</td>
      <td>net worth mortgages economist survey of consum...</td>
      <td>Wealth Mortgage loan Economist Survey of Consu...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>http://www.cnn.com/2010/SHOWBIZ/celebrity.news...</td>
      <td>3517403661074433</td>
      <td>68520890</td>
      <td>2010-11-13 18:39:49 UTC</td>
      <td>48949</td>
      <td>wheelchair Toulouse-Lautrec Los Angeles, Calif...</td>
      <td>Wheelchair Henri de Toulouse-Lautrec Los Angel...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>http://y.ahoo.it/FzXGKS</td>
      <td>6629109681623040</td>
      <td>113850982</td>
      <td>2010-11-22 08:44:37 UTC</td>
      <td>52814</td>
      <td>friends. You Riyadh Google Groups amd no free ...</td>
      <td>FriendsWithYou Riyadh Google Groups Advanced M...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>http://on.cnn.com/ic2iYo</td>
      <td>6940757323677696</td>
      <td>759251</td>
      <td>2010-11-23 05:23:00 UTC</td>
      <td>50693</td>
      <td>thing. I Mitt Romney Massachusetts George Bush...</td>
      <td>Treehouse of Horror VII Mitt Romney Massachuse...</td>
    </tr>
  </tbody>
</table>
</div>




```python
newsUserId = dfGoldStandard.userid.values.tolist()
newsUrl = dfGoldStandard.url.values.tolist()
newsId = dfGoldStandard.NewsId.values.tolist()

#dictionary of users who posted a newsArticle
newsId2UserId = {}

for i in range(len(newsId)):
    if newsId[i] not in newsId2UserId:
        newsId2UserId[newsId[i]] = []
    newsId2UserId[newsId[i]].append(newsUserId[i])
```


```python
# loading another dataframe with goldstandard but only keeping unique newsids
dfUniqueNewsId = pd.read_csv('../csvfiles/GoldStandard.csv',  lineterminator='\n', low_memory=False)
dfUniqueNewsId.drop_duplicates(subset='NewsId', inplace = True)
newsArticles = dfUniqueNewsId.NewsConceptText.values.tolist()
```


```python
#storing words in news articles in a list
newsArticlesForCorpus = [x.split(' ') for x in newsArticles]
#creating a corpus
newsId2word = corpora.Dictionary(newsArticlesForCorpus)
NewsArticlesCorpus = [newsId2word.doc2bow(text) for text in newsArticlesForCorpus]

#using the previous lda_model with the news corpus created to get a percent contribution for each topic for each news article
TopicDistributionOnNewsArticles = lda_model[NewsArticlesCorpus]
```

#### Creating User Vectors of length K where K is number of topics


```python
ArticleVector = []

for row in TopicDistributionOnNewsArticles:
    temp = [0]*topicNum
    for val in row:
        #val is a tuple in form (topicNum, percentContributionOfTopicToUser)
        temp[val[0]] = val[1]
    ArticleVector.append(temp)
    
print("Displaying sample article vector")
print(ArticleVector[1])
```

    Displaying sample article vector
    [0, 0, 0, 0, 0, 0.097441904, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07318355, 0, 0, 0, 0, 0, 0.16721845, 0, 0.17417727, 0, 0, 0, 0.17033984, 0, 0, 0, 0, 0, 0, 0.12391879, 0, 0, 0, 0, 0.1010313, 0, 0, 0.059913788, 0]


## Clustering

### Storing and preloading Kmeans results


```python
#different cluster sizes to try out analysis for
numClusters=[5, 10, 15, 20, 25, 30]
today = datetime.datetime.now()

#saving kmeans results for the differnt cluster sizes
for x in range(len(numClusters)):
    userVectorsFit = np.array(UserVectors)
    #performing kmeans on the userVector to cluster users into communities
    kmeans = KMeans(n_clusters=numClusters[x], random_state=0).fit(userVectorsFit)
    
    kMeansfilename = 'kMeans'+ today.strftime("%M%d") + 'CSize' + str(numClusters[x])
    pickle.dump(kmeans, open("../kmeansFiles/" + kMeansfilename,'wb'))
```


```python
#change this number to a number from the [5, 10, 15, 20, 25, 30] to preload a different file
chosenNumberOfCluster = 30

#loading existing kmeans model
kMeansfilename = 'kMeans' + today.strftime("%M%d") + 'CSize' + str(chosenNumberOfCluster)
print('Chosen File: \''+kMeansfilename+'\'')

loadedKmeansModel = pickle.load(open("../kmeansFiles/" + kMeansfilename, 'rb'))

```

    Chosen File: 'kMeans0604CSize30'


### Number of Users in each Cluster



```python
#creating a list to show how many users are in each cluster
userClusters = [0]*chosenNumberOfCluster
for i in loadedKmeansModel.labels_:
    userClusters[i] += 1

print(userClusters)
```

    [2602, 2461, 2731, 1926, 4811, 2063, 3690, 4688, 3347, 4461, 3819, 3963, 19409, 2272, 6060, 2086, 6527, 2873, 3048, 1810, 1921, 4678, 1703, 4297, 1405, 1358, 2245, 2710, 2191, 5063]


### User Indexes in each cluster, organized as an array



```python
UserIndexInCluster=[]
idsDict = list(dictConcept.keys())

for x in range(chosenNumberOfCluster):
    UserIndexInCluster.append([])
    
for index, val in enumerate(loadedKmeansModel.labels_):
    UserIndexInCluster[val].append(index)
```

### User ***IDs*** in each cluster, organized as an array



```python
idsCluster = []
for x in range(chosenNumberOfCluster):
    idsCluster.append([])
    
for index, val in enumerate(loadedKmeansModel.labels_):
    idsCluster[val].append(idsDict[index])   
```

### Find Topic Distribution Per Cluster


```python
topicDistributionPerCluster=[]
for x in range(chosenNumberOfCluster):
    topicDistributionPerCluster.append([])
    
for i,cluster in enumerate(UserIndexInCluster):
    for userIndex in cluster:
        topicDistributionPerCluster[i].append(UserVectors[userIndex])
```

### Find Average Topic distribution per Cluster



```python
averageDistributionPerCluster = []
for x in topicDistributionPerCluster:
    y = np.array(x)
    listOfAverageValues = np.mean(y,axis=0)
    averageDistributionPerCluster.append(listOfAverageValues)
print(listOfAverageValues)
```

    [0.00682627 0.00611497 0.00515724 0.00504671 0.01006376 0.02487488
     0.00399558 0.00511773 0.00365652 0.01216871 0.00479767 0.0043347
     0.00329815 0.00427418 0.00690943 0.00282821 0.01156971 0.02049791
     0.00293984 0.00710016 0.00380569 0.00555181 0.00374264 0.00732811
     0.00915575 0.00436225 0.0048657  0.00204496 0.00415432 0.00260818
     0.01303746 0.582369   0.00513088 0.0033796  0.00371658 0.00315607
     0.00293613 0.00284987 0.00461186 0.00313167 0.02004369 0.00193077
     0.00571504 0.00293978 0.00741164 0.00316571 0.00492713]


#### Ranking Articles to a Cluster


```python
from scipy import spatial

rankArticlesToCluster=[]
for x in range(chosenNumberOfCluster):
    rankArticlesToCluster.append([])
    
for x in range (len(ArticleVector)):
    for index,value in enumerate(averageDistributionPerCluster):
        #finds cosine similarity between artlice vector and average vector of the cluster
        rankArticlesToCluster[index].append(tuple((x,1 - spatial.distance.cosine(ArticleVector[x], value))))
        
#sorting the ranked list
import operator
sortedRankArticlesToCluster=[]
for x in rankArticlesToCluster:
    sortedRankArticlesToCluster.append(sorted(x,key=lambda x: x[1]))

ascendingRankedArticlesToCluster = []
for x in sortedRankArticlesToCluster:
    ascendingRankedArticlesToCluster.append(list(reversed(x)))
        

```

#### Ranking Clusters to an Article


```python
rankClustersToArticle = []
for x in range(len(ArticleVector)):
    rankClustersToArticle.append([])
    
for x in range(chosenNumberOfCluster):
    for index, value in enumerate(ArticleVector):
        rankClustersToArticle[index].append(tuple((x, 1-spatial.distance.cosine(value, averageDistributionPerCluster[x]))))
        
#sorting the ranked list

sortedRankClustersToArticle=[]
for x in rankClustersToArticle:
    sortedRankClustersToArticle.append(sorted(x,key=lambda x: x[1]))

ascendingRankClustersToArticle = []
for x in sortedRankClustersToArticle:
    ascendingRankClustersToArticle.append(list(reversed(x)))
```

## Metrics and Evaluation

### News Recommendation

#### S@10 Version 1 where we compare if one user who posted the aricle exists in the community


```python
def sAt10OneUser():
    k=10
    total=0
    for x in range(len(ascendingRankedArticlesToCluster)):
        count = 0;
        for y in ascendingRankedArticlesToCluster[x][:k]:
            newsid = int(dfUniqueNewsId.iloc[[y[0]]].NewsId)
            for user in newsId2UserId[newsid]:
                if user in idsCluster[x]:
                    count+=10
                    total+=10
                    break
            if count != 0:
                break
    precisionVal = total/(chosenNumberOfCluster*10)
    print(precisionVal)
```

#### S@10 Version 2 where we compare if all users who posted the aricle exists in the community



```python
def sAt10AllUsers():
    k=10
    total=0
    for x in range(len(ascendingRankedArticlesToCluster)):
        count = 0;
        for y in ascendingRankedArticlesToCluster[x][:k]:
            newsid = int(dfUniqueNewsId.iloc[[y[0]]].NewsId)
            if len(set(newsId2UserId[newsid])&set(idsCluster[x])) == len(newsId2UserId[newsid]):
                count += 10
                total+=10
                break
    precisionVal = total/(chosenNumberOfCluster*10)
    print(precisionVal)
```


```python
sAt10OneUser()
sAt10AllUsers()
```

    0.36666666666666664
    0.06666666666666667


#### MRR Version 1 where we compare if one user who posted the aricle exists in the community


```python
def mrrOneUser():
    mrr=0
    totalmrr = 0
    for x in range(len(ascendingRankedArticlesToCluster)):
        mrr = 0
        for index, y in enumerate(ascendingRankedArticlesToCluster[x]):
            newsid = int(dfUniqueNewsId.iloc[[y[0]]].NewsId)
            for user in newsId2UserId[newsid]:
                if user in idsCluster[x]:
                    mrr= (1/(index + 1))
                    totalmrr += mrr
                    break
            if(mrr != 0):
                break
    print(totalmrr/chosenNumberOfCluster)
```

#### MRR Version 2 where we compare if all users who posted the aricle exists in the community



```python
def mrrAllUsers():
    mrr=0
    totalmrr = 0
    for x in range(len(ascendingRankedArticlesToCluster)):
        mrr = 0
        for index, y in enumerate(ascendingRankedArticlesToCluster[x]):
            newsid = int(dfUniqueNewsId.iloc[[y[0]]].NewsId)
            if len(set(newsId2UserId[newsid])&set(idsCluster[x])) == len(newsId2UserId[newsid]):
                mrr= (1/(index + 1))
                totalmrr += mrr
                break
    print(totalmrr/chosenNumberOfCluster)
```


```python
mrrOneUser()
mrrAllUsers()
```

    0.21833547028700867
    0.029922369836447076


### User Prediction

#### Precision and Recall


```python
NewsIdsKeys = list(newsId2UserId.keys())
```


```python
def precisionAndRecall():
    fn = 0
    tp = 0
    fp = 0
    precision = 0
    recall = 0
    
    for index, val in enumerate(NewsIdsKeys):
        fp = 0
        tp = 0
        fn = 0
        c = ascendingRankClustersToArticle[index][0][0]
        tp = len(set(newsId2UserId[val])&set(idsCluster[c]))
        fp = (len(idsCluster[c]) - tp)
        fn = (len(newsId2UserId[val]) - tp)

        if (tp+fp)!=0:
            precision = precision + tp/(tp+fp)
        if (tp+fn)!=0:
            recall = recall + tp/(tp+fn)
    overallPrecision = precision/len(NewsIdsKeys)
    overallRecall = recall/len(NewsIdsKeys)
    return (overallPrecision, overallRecall)
```


```python
precisionAndRecall()
```




    (5.960150378862192e-05, 0.06152784406574581)



#### FMeasure


```python
x=precisionAndRecall()
fmeasure= 2*((x[0]*x[1])/(x[0]+x[1]))
print(fmeasure)
```

    0.00011908764837632181


#### TODO: STORE FINAL RESULTS FOR DIFF VALUES IN CSV AND SHOW TABLE 


```python

```


```python

```
