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

from nrclex import NRCLex
import re
from gensim.parsing.preprocessing import STOPWORDS, strip_numeric, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short
import en_core_web_sm
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
    
def create_emotion_wordclouds(text):
    clean_sentences = _preprocess_text(text)
    emotion_dictionary = _create_emotion_dictionary(clean_sentences)
    _plot_wordclouds(emotion_dictionary)

def _preprocess_text(text):
    clean_sentences = []
    nlp = en_core_web_sm.load()
    sentences = text.split('.')
    for sentence in sentences: 
        no_numbers = strip_numeric(sentence)
        no_punctuation = strip_punctuation(no_numbers)
        no_extra_whitespaces = strip_multiple_whitespaces(no_punctuation)
        stripped = no_extra_whitespaces.strip()
        lowercase = stripped.lower()
        no_stopwords = remove_stopwords(lowercase)
        no_short_words = strip_short(no_stopwords)
        lemmatized = ' '.join([token.lemma_ for token in nlp(no_short_words)])
        if len(lemmatized) > 0:
            clean_sentences.append(lemmatized)
    return clean_sentences

def _create_emotion_dictionary(sentences):
    emotion_dictionary = {}
    for sentence in sentences:
        text_object = NRCLex(sentence)
        if len(text_object.affect_list) > 0:
            top_emotion = text_object.top_emotions[0][0]
            if top_emotion in emotion_dictionary:
                emotion_dictionary[top_emotion] += ' {}'.format(sentence)
            else:
                emotion_dictionary[top_emotion] = sentence
    return emotion_dictionary

def _plot_wordclouds(emotion_dictionary):
    for key, value in emotion_dictionary.items():
        plt.figure()
        plt.imshow(WordCloud(background_color='white', width=600, height=300).generate(value))
        plt.axis("off")
        plt.title(key)
        plt.show()
