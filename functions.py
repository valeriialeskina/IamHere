'''
Instructions/Notes
Various functions and libraries for the project are imported or defined in this file, to avoid a lot of information on the project file.
'''
import pandas as pd
import numpy as np
import re
from gensim.parsing.preprocessing import STOPWORDS, strip_tags, strip_numeric, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, stem_text
import pickle
import spacy # import en_core_web_sm 
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')

import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from nltk.tokenize import word_tokenize

nltk.download('punkt')
import demoji
demoji.download_codes()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def clean_complete(tweet): #to be adjusted for the diary -- for now it is copypasted for tweets
    """
    tweet: pandas series
    prepares tweets complete cleaning for further lemmatization and dering embeddings
    """
    pat = r"(\\n)|(@\w*)|((www\.[^\s]+)|(https?://[^\s]+))"
    tweet = tweet.str.replace(pat, '')

    #remove repeated charachters
    
    #replace emoticons with words
    #SMILEYS = {":-(":"sad", ":‑)":"smiley", ":-P":"playfullness", ":-/":'confused'}

    tweet = tweet.str.replace(r':-\)', ' smile')
    tweet = tweet.str.replace(r':-\(', ' sad')
    tweet = tweet.str.replace(r':-\/', ' confused')
    tweet = tweet.str.replace(r':-P', ' playfullness')

    #delete \xa
    tweet = tweet.str.replace('\xa0', '')

    tweet = tweet.str.replace('&amp', '')
    tweet = tweet.str.replace('\n', '')
    tweet = tweet.str.replace('"', '')
    #to lower case
    tweet = tweet.str.lower()

    #covert hashtags to the normal text
    tweet = tweet.str.replace(r'#([^\s]+)', r'\1')

    #delete numbers
    tweet = [strip_numeric(c) for c in tweet]

    #replacing emojies with descriptions '❤️-> red heart'
    tweet = [demoji.replace_with_desc(c, ' ') for c in tweet]

    #delete punctuation
    tweet = [strip_punctuation(c) for c in tweet]

    #remove stop words
    tweet = [remove_stopwords(c) for c in tweet]

    #remove short words
    tweet = [strip_short(c) for c in tweet]

    #remove mult whitespaces
    tweet = [strip_multiple_whitespaces(c) for c in tweet]
    return tweet

def clean_vader(tweet): #to be adjusted for the diary -- for now it is copypasted for tweets
    """
    tweet: pandas series
    prepares tweets for vader sentiment analysis
    """

    pat = r"(\\n)|(@\w*)|((www\.[^\s]+)|(https?://[^\s]+))"
    tweet = tweet.str.replace(pat, '')

    #replace emoticons with words
    #SMILEYS = {":-(":"sad", ":‑)":"smiley", ":-P":"playfullness", ":-/":'confused'}

    #tweet = tweet.str.replace(r':-\)', ' smile')
    #tweet = tweet.str.replace(r':-\(', ' sad')
    #tweet = tweet.str.replace(r':-\/', ' confused')
    #tweet = tweet.str.replace(r':-P', ' playfullness')

    #delete \xa
    tweet = tweet.str.replace('\xa0', '')

    tweet = tweet.str.replace('&amp', '')
    tweet = tweet.str.replace('\n', '')

    #to lower case
    #tweet = tweet.str.lower()

    #covert hashtags to the normal text
    tweet = tweet.str.replace(r'#([^\s]+)', r'\1')

    #delete numbers
    tweet = [strip_numeric(c) for c in tweet]

    #replacing emojies with descriptions '❤️-> red heart'
    #tweet = [demoji.replace_with_desc(c, ' ') for c in tweet]

    #delete punctuation
    #tweet = [strip_punctuation(c) for c in tweet]

    #remove stop words
    #tweet = [remove_stopwords(c) for c in tweet]

    #remove short words
    tweet = [strip_short(c) for c in tweet]

    #remove mult whitespaces
    tweet = [strip_multiple_whitespaces(c) for c in tweet]
    return tweet

def lemmatize(tweet):
    '''
    tweet: pandas series
    should be applied on the cleaned tweets to transform words to their initial base form.
    For example: suggests -> suggest, deliveries -> delivery
    '''
    nlp = spacy.load("en_core_web_sm")
    tweet = [nlp(c) for c in tweet]
    tweet = [" ".join([token.lemma_ for token in t]) for t in tweet]
    return tweet
