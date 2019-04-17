
# **Capstone EB04**

The following version of the code is for finding like minded user communities by getting higher level topics during the lda phase before performing clustering. This is the second of two different methods that were attempted.

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
#stores a list of each tweet and the words the tweets contain
tempData = []

for sent in tweetConcept:
    x = []
    for word in sent.split(" "):
        x.append(word)
    tempData.append(x)

data_final = tempData
```

## LDA Analysis


```python
#setting up corpus for lda
id2word1 = corpora.Dictionary(data_final)
texts = data_final
corpus1 = [id2word1.doc2bow(text) for text in texts]
```

### Run LDA
Multicore allows for multiple cores to be working on LDA simultaneously
- Check Number of workers
- Check Number of topics set<br>


```python
#uncomment line below to try lda with different values
topicNum = 47
#lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus1,id2word=id2word1,num_topics=topicNum, passes=10, workers=7)
#lda_model.save('../LdaSaves/topics47mar10p9/lda.model_mar10_t47')

#preloading an saved lda run to save time as lda takes long time to run
lda_model1 =  models.LdaModel.load('../LDASaves/ldamar25/lda.model_mar25_t47')
pprint(lda_model1.print_topics())
```

    [(35,
      '0.139*"Thanksgiving" + 0.042*"Turkey" + 0.030*"Unemployment" + 0.021*"Star" '
      '+ 0.021*"Mark" + 0.021*"market" + 0.021*"(bird)" + 0.020*"Foreign" + '
      '0.018*"Pour" + 0.018*"Jalili"'),
     (1,
      '0.241*"New" + 0.096*"York" + 0.060*"Day" + 0.059*"International" + '
      '0.055*"Year\'s" + 0.048*"Airport" + 0.046*"City" + 0.024*"Eve" + '
      '0.023*"Jersey" + 0.018*"of"'),
     (33,
      '0.050*"San" + 0.034*"Francisco" + 0.030*"Climate" + 0.024*"with" + '
      '0.024*"For" + 0.023*"Australia" + 0.022*"Family" + 0.021*"Snow" + '
      '0.019*"Economist" + 0.019*"Transmitter"'),
     (29,
      '0.123*"California" + 0.055*"Service" + 0.037*"Africa" + 0.034*"ITunes" + '
      '0.026*"Atlanta" + 0.025*"Group" + 0.025*"Americans" + 0.022*"Georgia" + '
      '0.019*"television" + 0.018*"Now"'),
     (23,
      '0.187*"the" + 0.127*"of" + 0.107*"United" + 0.082*"States" + 0.063*"in" + '
      '0.031*"Israel" + 0.028*"Kingdom" + 0.023*"President" + 0.015*"Press" + '
      '0.011*"and"'),
     (34,
      '0.121*"(film)" + 0.051*"Palin" + 0.050*"Sarah" + 0.025*"Brazil" + '
      '0.023*"Alaska" + 0.023*"Green" + 0.023*"service" + 0.023*"Man" + '
      '0.018*"Wind" + 0.016*"Lee"'),
     (25,
      '0.062*"football)" + 0.059*"(American" + 0.050*"CNN" + 0.049*"Federal" + '
      '0.043*"(magazine)" + 0.036*"Tackle" + 0.026*"Commission" + '
      '0.023*"Communications" + 0.022*"Spanish" + 0.018*"System"'),
     (27,
      '0.172*"United" + 0.144*"States" + 0.130*"WikiLeaks" + 0.062*"RT" + '
      '0.062*"network)" + 0.060*"(TV" + 0.033*"Senate" + 0.032*"Julian" + '
      '0.031*"Assange" + 0.029*"leak"'),
     (45,
      '0.106*"of" + 0.046*"List" + 0.044*"Security" + 0.041*"La" + '
      '0.032*"Department" + 0.025*"Court" + 0.025*"Latin-script" + '
      '0.025*"digraphs" + 0.022*"King" + 0.021*"Administration"'),
     (20,
      '0.070*"Time" + 0.066*"State" + 0.061*"Mexico" + 0.058*"film)" + '
      '0.039*"Central" + 0.034*"Agency" + 0.024*"of" + 0.024*"Home" + 0.021*"NATO" '
      '+ 0.020*"(2012"'),
     (28,
      '0.119*"Obama" + 0.113*"Barack" + 0.045*"of" + 0.039*"Japan" + '
      '0.032*"France" + 0.021*"French" + 0.020*"Video" + 0.019*"Boston" + '
      '0.018*"Science" + 0.018*"Photography"'),
     (39,
      '0.045*"music" + 0.034*"Email" + 0.033*"(computing)" + 0.032*"Venezuela" + '
      '0.025*"I\'m" + 0.020*"Mashable" + 0.019*"Suicide" + 0.018*"Stock" + '
      '0.016*"Rome" + 0.013*"hop"'),
     (44,
      '0.065*"YouTube" + 0.056*"Show" + 0.047*"Europe" + 0.039*"for" + '
      '0.034*"Cancer" + 0.030*"Computer" + 0.021*"All" + 0.020*"Early" + '
      '0.019*"Song" + 0.017*"Orleans"'),
     (30,
      '0.075*"ATP" + 0.062*"&amp;" + 0.031*"Chris" + 0.030*"Muslim" + '
      '0.023*"Islam" + 0.022*"Disney" + 0.020*"IPod" + 0.019*"Walt" + '
      '0.018*"Johnson" + 0.018*"Lawsuit"'),
     (17,
      '0.101*"Afghanistan" + 0.086*"War" + 0.030*"in" + 0.028*"Law" + 0.021*"of" + '
      '0.019*"Society" + 0.018*"High" + 0.017*"Switzerland" + '
      '0.016*"(2001–present)" + 0.016*"Photograph"'),
     (7,
      '0.158*"China" + 0.136*"Facebook" + 0.066*"Pakistan" + 0.033*"Daily" + '
      '0.026*"officer" + 0.021*"Chief" + 0.019*"News" + 0.019*"Energy" + '
      '0.017*"Information" + 0.016*"Solar"'),
     (11,
      '0.077*"Police" + 0.032*"Your" + 0.026*"Red" + 0.025*"Winter" + '
      '0.023*"County," + 0.021*"Foundation" + 0.021*"Chinese" + 0.016*"(music)" + '
      '0.015*"Welfare" + 0.014*"–"'),
     (31,
      '0.086*"(band)" + 0.072*"to" + 0.053*"In" + 0.048*"Tax" + 0.044*"(album)" + '
      '0.043*"Me" + 0.040*"de" + 0.019*"cut" + 0.018*"Club" + 0.016*"Paulo"'),
     (36,
      '0.054*"Health" + 0.049*"Los" + 0.044*"Angeles" + 0.034*"Terrorism" + '
      '0.034*"Food" + 0.024*"Drug" + 0.021*"Zealand" + 0.019*"care" + '
      '0.018*"Cannabis" + 0.018*"and"'),
     (10,
      '0.583*"Twitter" + 0.024*"software" + 0.019*"Application" + 0.014*"Steve" + '
      '0.011*"and" + 0.011*"Patient" + 0.010*"Care" + 0.009*"Protection" + '
      '0.009*"Act" + 0.009*"Affordable"')]



```python
#creating a dictionary where each key is the user and the value is a list of all topicsNums representing each tweet
dictConcept = {}

for i in range(len(userIds)):
    if userIds[i] not in dictConcept:
        dictConcept[userIds[i]] = []

#getting the top 3 topics for each user tweet and appending to the user dictionary
for i, row in enumerate (lda_model1[corpus1]):
    sortedValue = sorted(row, key=lambda x:x[1], reverse=True)
    userid =int(df.iloc[[i]].userid)
    for z in sortedValue[:3]:
        dictConcept[userid].append(str(z[0]))
```

### Running 2nd Round of LDA To find Higher Level Topics


```python
import ast
#using previously saved list to save time
# topicsPerTweets=[]
# with open('topicsPerTweets.txt', 'r') as f:
#     for line in f:
#         topicsPerTweets.append(ast.literal_eval(line))
        
topicsPerTweets = list(dictConcept.values())
print(topicsPerTweets[:2])

id2word2 = corpora.Dictionary(topicsPerTweets)
corpus2 = [id2word2.doc2bow(text) for text in topicsPerTweets]

#saving the list so the above block does not need to be rerun since it takes a while
# with open('topicsPerTweets.txt', 'w') as f:
#     for item in topicsPerTweets:
#         f.write("%s\n" % item)
```

    [['8', '23', '41'], ['33', '0', '1']]



```python
topicNum2 = 10
today = datetime.datetime.now()
#uncomment the line below to run with own custom topics numbers or workers
lda_model2 = gensim.models.ldamulticore.LdaMulticore(corpus=corpus2, id2word=id2word2, num_topics=topicNum2, passes=10, workers=3)
lda_model2.save('../LDASaves/HigherOrderModels/LDA' + today.strftime("%M%d") + str(topicNum2))

lda_model2 =  models.LdaModel.load('../LDASaves/HigherOrderModels/LDA' + today.strftime("%M%d") + str(topicNum2))
doc_lda2 = lda_model2[corpus2]
pprint(lda_model2.print_topics())

```

    [(0,
      '0.131*"7" + 0.111*"38" + 0.093*"14" + 0.074*"16" + 0.054*"39" + 0.050*"24" '
      '+ 0.040*"1" + 0.036*"26" + 0.031*"0" + 0.025*"5"'),
     (1,
      '0.093*"6" + 0.086*"33" + 0.085*"9" + 0.076*"36" + 0.069*"12" + 0.060*"44" + '
      '0.052*"35" + 0.049*"1" + 0.042*"0" + 0.031*"3"'),
     (2,
      '0.128*"8" + 0.105*"0" + 0.105*"4" + 0.097*"26" + 0.088*"1" + 0.050*"37" + '
      '0.039*"12" + 0.030*"31" + 0.021*"38" + 0.020*"23"'),
     (3,
      '0.195*"15" + 0.169*"25" + 0.085*"29" + 0.039*"22" + 0.038*"1" + 0.033*"3" + '
      '0.025*"30" + 0.024*"2" + 0.024*"32" + 0.022*"0"'),
     (4,
      '0.329*"21" + 0.189*"18" + 0.049*"1" + 0.045*"0" + 0.033*"44" + 0.031*"37" + '
      '0.031*"12" + 0.019*"40" + 0.018*"6" + 0.018*"23"'),
     (5,
      '0.579*"10" + 0.066*"30" + 0.057*"4" + 0.052*"26" + 0.052*"3" + 0.050*"21" + '
      '0.038*"0" + 0.033*"1" + 0.015*"25" + 0.014*"46"'),
     (6,
      '0.187*"27" + 0.069*"5" + 0.061*"32" + 0.056*"23" + 0.046*"1" + 0.046*"6" + '
      '0.044*"20" + 0.041*"17" + 0.032*"0" + 0.030*"7"'),
     (7,
      '0.322*"1" + 0.314*"0" + 0.124*"2" + 0.017*"31" + 0.012*"39" + 0.012*"45" + '
      '0.011*"35" + 0.011*"14" + 0.010*"28" + 0.009*"5"'),
     (8,
      '0.096*"41" + 0.091*"40" + 0.086*"11" + 0.083*"1" + 0.077*"24" + 0.071*"31" '
      '+ 0.068*"19" + 0.055*"0" + 0.032*"8" + 0.028*"33"'),
     (9,
      '0.119*"46" + 0.114*"23" + 0.103*"12" + 0.069*"28" + 0.066*"42" + 0.059*"43" '
      '+ 0.045*"34" + 0.033*"32" + 0.024*"13" + 0.021*"35"')]


#### Compute Perplexity and Coherence


```python
print('\nPerplexity: ', lda_model2.log_perplexity(corpus2))  # a measure of how good the model is. lower the better.
coherence_model_lda = CoherenceModel(model=lda_model2, texts=topicsPerTweets, dictionary=id2word2, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
```

    
    Perplexity:  -3.5847416341948763
    
    Coherence Score:  0.27723398824434303


#### Finding Topic Distribution


```python
def format_topics_sentences(ldamodel=lda_model2, corpus=corpus2, texts=topicsPerTweets):
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


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model2, corpus=corpus2, texts=topicsPerTweets)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
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
      <th>Document_No</th>
      <th>Dominant_Topic</th>
      <th>Topic_Perc_Contrib</th>
      <th>Keywords</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>8.0</td>
      <td>0.5658</td>
      <td>41, 40, 11, 1, 24, 31, 19, 0, 8, 33</td>
      <td>[8, 23, 41]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.0</td>
      <td>0.5231</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>[33, 0, 1]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.2938</td>
      <td>7, 38, 14, 16, 39, 24, 1, 26, 0, 5</td>
      <td>[29, 16, 10]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.0</td>
      <td>0.7750</td>
      <td>6, 33, 9, 36, 12, 44, 35, 1, 0, 3</td>
      <td>[12, 13, 3]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>9.0</td>
      <td>0.4757</td>
      <td>46, 23, 12, 28, 42, 43, 34, 32, 13, 35</td>
      <td>[23, 30, 38]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>7.0</td>
      <td>0.3070</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>[40, 33, 33, 27, 27, 17, 8, 3, 30, 0, 1, 28, 1...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7.0</td>
      <td>0.3862</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>[10, 18, 2, 28, 30, 25, 0, 1, 0, 1, 2, 15, 34,...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>3.0</td>
      <td>0.4257</td>
      <td>15, 25, 29, 22, 1, 3, 30, 2, 32, 0</td>
      <td>[21, 10, 6, 22, 29]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>7.0</td>
      <td>0.8306</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>[40, 0, 1, 35, 0, 1, 16, 0, 1, 31, 0, 1]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>7.0</td>
      <td>0.2713</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>[31, 30, 4, 33, 25, 10, 11, 26, 16, 0, 1, 2, 2...</td>
    </tr>
  </tbody>
</table>
</div>




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
df_dominant_topics[0:10]
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
      <th>Dominant_Topic</th>
      <th>Topic_Keywords</th>
      <th>Num_Documents</th>
      <th>Perc_Documents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0</td>
      <td>41, 40, 11, 1, 24, 31, 19, 0, 8, 33</td>
      <td>9557.0</td>
      <td>0.0852</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>11226.0</td>
      <td>0.1000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>7, 38, 14, 16, 39, 24, 1, 26, 0, 5</td>
      <td>10876.0</td>
      <td>0.0969</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>6, 33, 9, 36, 12, 44, 35, 1, 0, 3</td>
      <td>7730.0</td>
      <td>0.0689</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.0</td>
      <td>46, 23, 12, 28, 42, 43, 34, 32, 13, 35</td>
      <td>3877.0</td>
      <td>0.0345</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7.0</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>1960.0</td>
      <td>0.0175</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>9386.0</td>
      <td>0.0836</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.0</td>
      <td>15, 25, 29, 22, 1, 3, 30, 2, 32, 0</td>
      <td>33482.0</td>
      <td>0.2984</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.0</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>11827.0</td>
      <td>0.1054</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7.0</td>
      <td>1, 0, 2, 31, 39, 45, 35, 14, 28, 5</td>
      <td>12297.0</td>
      <td>0.1096</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating User Vectors of length K where K is number of topics


```python
UserVectors = []

#for each users shows percent contribution for that topic
print(lda_model2[corpus2][1][1])

for row in lda_model2[corpus2]:
    temp = [0]*topicNum2
    for val in row:
        #val is a tuple in form (topicNum, percentContributionOfTopicToUser)
        temp[val[0]] = val[1]
    UserVectors.append(temp)
    
print("Shows a sample userVector")    
print(UserVectors[1])
```

    (1, 0.2756533)
    Shows a sample userVector
    [0.025003769, 0.27573124, 0.025003202, 0.025001932, 0.025001537, 0.025001056, 0.02500354, 0.52424777, 0.025005233, 0.025000773]


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

### Getting 1st Round LDA topics for news articles


```python
#storing words in news articles in a list
newsArticlesForCorpus = [x.split(' ') for x in newsArticles]
#creating a corpus
newsId2word = corpora.Dictionary(newsArticlesForCorpus)
NewsArticlesCorpus = [newsId2word.doc2bow(text) for text in newsArticlesForCorpus]

#using the previous lda_model for first lda run with the news corpus created to get a percent contribution for each topic for each news article
TopicDistributionOnNewsArticles = lda_model1[NewsArticlesCorpus]
```


```python
# finding the top 3 topics for each news article
topicsPerNewsArticleHighLevel = []
for x in (TopicDistributionOnNewsArticles):
    sortedValue = sorted(x, key=lambda x:x[1], reverse=True)
    temp = []
    for z in sortedValue[:3]:
        temp.append(str(z[0]))
    topicsPerNewsArticleHighLevel.append(temp)

print(topicsPerNewsArticleHighLevel[:5])
```

    [['34', '8', '10'], ['23', '22', '46'], ['4', '8', '15'], ['0', '24', '8'], ['32', '44', '25']]


### Getting 2nd Round LDA topics (High Level Topics) for news articles


```python
#setting up corpus for 2nd lda run on news articles
newsId2word2 = corpora.Dictionary(topicsPerNewsArticleHighLevel)
NewsArticlesCorpus2 = [newsId2word2.doc2bow(text) for text in topicsPerNewsArticleHighLevel]
TopicDistributionOnNewsArticles = lda_model2[NewsArticlesCorpus2]
```

#### Creating User Vectors of length K where K is number of topics


```python
ArticleVector = []

for row in TopicDistributionOnNewsArticles:
    temp = [0]*topicNum2
    for val in row:
        #val is a tuple in form (topicNum, percentContributionOfTopicToUser)
        temp[val[0]] = val[1]
    ArticleVector.append(temp)
    
print("Displaying sample article vector")
print(ArticleVector[1])
```

    Displaying sample article vector
    [0.025003776, 0.27553427, 0.025003204, 0.025001936, 0.025001539, 0.025001058, 0.025003547, 0.5244446, 0.025005287, 0.025000777]


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
    
    kMeansfilename = 'LDAM2-kMeans'+ today.strftime("%M%d") + 'CSize' + str(numClusters[x])
    pickle.dump(kmeans, open("../kmeansFiles/" + kMeansfilename,'wb'))
```


```python
#change this number to a number from the [5, 10, 15, 20, 25, 30] to preload a different file
chosenNumberOfCluster = 30

#loading existing kmeans model
kMeansfilename = 'LDAM2-kMeans' + today.strftime("%M%d") + 'CSize' + str(chosenNumberOfCluster)
print('Chosen File: \''+kMeansfilename+'\'')

loadedKmeansModel = pickle.load(open("../kmeansFiles/" + kMeansfilename, 'rb'))

```

    Chosen File: 'LDAM2-kMeans1404CSize30'


### Number of Users in each Cluster



```python
#creating a list to show how many users are in each cluster
userClusters = [0]*chosenNumberOfCluster
for i in loadedKmeansModel.labels_:
    userClusters[i] += 1

print(userClusters)
```

    [12453, 4082, 1667, 6022, 1458, 4969, 1229, 2526, 3356, 3832, 2846, 4338, 3842, 4305, 3366, 2048, 4078, 2253, 5937, 4338, 3840, 3749, 3780, 4793, 2469, 3621, 1345, 4223, 1506, 3947]


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

    [0.03202455 0.03888813 0.32654442 0.03580891 0.02753587 0.02751216
     0.02991044 0.39240489 0.04464912 0.03473951]


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

    0.4666666666666667
    0.03333333333333333


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

    0.18742772866109633
    0.04276173585261982


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




    (3.6714102128504575e-05, 0.025113231739025306)



#### FMeasure


```python
x=precisionAndRecall()
fmeasure= 2*((x[0]*x[1])/(x[0]+x[1]))
print(fmeasure)
```

    7.332101314784211e-05


#### TODO: STORE FINAL RESULTS FOR DIFF VALUES IN CSV AND SHOW TABLE 


```python

```


```python

```
