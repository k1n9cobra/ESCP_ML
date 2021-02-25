## NLP Topic detection from urls

import os
import json
import pandas as pd
import string
import re
import seaborn as sns
import numpy as np
import gensim
from gensim import corpora, models
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_numeric
import nltk

data = pd.read_csv('data_4_topicmodeling_1k.csv')
data.url_cleaned = data.url_cleaned.apply(lambda x : preprocess_string(x))

from nltk.corpus import stopwords
stopwords_other = ['www','com','fr','php','net','html', 'faq']
my_stopwords = stopwords.words('French') + stopwords_other

data.url_cleaned = data.url_cleaned.apply(lambda x : [token for token in x if token not in my_stopwords])

dictionary = corpora.Dictionary(data['url_cleaned'])
dictionary.filter_extremes(no_below=5,     
                           no_above=0.5, 
                           keep_n=100000)
corpus_bow = [dictionary.doc2bow(review) for review in data['url_cleaned']]

print('Number of unique tokens for training: {}'.format(len(dictionary)))
print('Number of documents for training: {}'.format(len(corpus_bow)))

num_topics = 6
random_state = 123
lda_model = models.LdaModel(corpus_bow, 
                            id2word=dictionary, 
                            num_topics=num_topics, 
                            random_state = random_state,
                            alpha = [0.01]*num_topics,
                            )

#alpha distribution of words in topics
#eta distribution of topics per document

for i, topic in lda_model.print_topics():
    print('Topic: {} \nWords: {}\n'.format(i, topic))

tfidf = models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 
                                             id2word=dictionary, 
                                             num_topics=num_topics,  
                                             random_state=random_state)
for i, topic in lda_model_tfidf.print_topics():
    print('Topic: {} \nWords: {}\n'.format(i, topic))
    
    
topics = []
perc_topics = []

for i in range(len(data)):
    topic = sorted(lda_model_tfidf[corpus_bow[i]],key=lambda tup: -1*tup[1])[0][0]
    perc_topic = sorted(lda_model_tfidf[corpus_bow[i]],key=lambda tup: -1*tup[1])[0][1]
    topics.append(topic)
    perc_topics.append(perc_topic)
data['topic'] = topics
data['topic_percentage'] = perc_topics

#Topic 0: Trintignan (acteur)
#Topic 1: meteo
#Topic 2: santé/symptomes
#Topic 3: gossip
#Topic 4: emploi/finance
#Topic 5: cuisine

topic_dict = {0:'Trintignan (acteur)',1:'météo', 2:'santé/symptomes',3:'gossip',4:'emploi/finance',5:'cuisine'}
data['topic'] = data['topic'].map(topic_dict)
data.to_csv('url_data_enriched_with_topics.csv')
