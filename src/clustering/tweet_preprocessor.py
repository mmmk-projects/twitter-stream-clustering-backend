import re

from langdetect import detect
from nltk import word_tokenize
from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import preprocessor as tweet

words = set(words.words())
stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def remove_links_and_hashtags(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row

    df[final_row] = df[init_row].apply(lambda text: tweet.clean(str(text)))

def to_lowercase(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row

    df[final_row] = df[init_row].apply(lambda text: text.lower())

def remove_contractions(df, final_row, init_row=None):
    def _remove_contractions(tweet):
        tweet = re.sub(r'’', '\'', tweet)
    
        tweet = re.sub(r'won\'t', 'will not', tweet)
        tweet = re.sub(r'can\'t', 'can not', tweet)
        
        tweet = re.sub(r'\'s', ' is', tweet)
        tweet = re.sub(r'\'m', ' am', tweet)
        tweet = re.sub(r'\'re', ' are', tweet)
        tweet = re.sub(r'\'ve', ' have', tweet)
        tweet = re.sub(r'\'ll', ' will', tweet)
        tweet = re.sub(r'\'d', ' would', tweet)
        tweet = re.sub(r'\'t', ' not', tweet)
        tweet = re.sub(r'n\'t', ' not', tweet)
        
        return tweet

    if init_row is None:
        init_row = final_row
    
    df[final_row] = df[init_row].apply(_remove_contractions)

def remove_punctuations(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row

    df[final_row] = df[init_row].str.replace(r'[^\w\s]', '')

def remove_whitespaces(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row
    
    df[final_row] = df[init_row].apply(lambda text: str(text).strip())

def remove_non_english_tweets(df, row):
    df['lang'] = df[row].apply(lambda text: detect(text))
    df.drop(df[df['lang'] != 'en'].index, inplace=True)
    df.drop(columns=['lang'], inplace=True)

def lemmatize(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row
    
    df[final_row] = df[init_row].apply(lambda text: ' '.join(lemmatizer.lemmatize(word)
                                                             for word in text.split()))

def remove_non_english_words(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row
    
    df[final_row] = df[init_row].apply(lambda text: ' '.join(word for word in text.split()
                                                             if word in words))

def remove_english_stopwords(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row
    
    df[final_row] = df[init_row].apply(lambda text: ' '.join(word for word in text.split()
                                                             if word not in stopwords))

def remove_common_words(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row
    
    word_counts = pd.Series(' '.join(df[init_row]).split()).value_counts()
    common_freq = word_counts[word_counts > 300]
    df[final_row] = df[init_row].apply(lambda text: ' '.join(word for word in text.split()
                                                             if word not in common_freq))

def remove_rare_words(df, final_row, init_row=None):
    if init_row is None:
        init_row = final_row
    
    word_counts = pd.Series(' '.join(df[init_row]).split()).value_counts()
    rare_freq = word_counts[word_counts <= 3]
    df[final_row] = df[init_row].apply(lambda text: ' '.join(word for word in text.split()
                                                             if word not in rare_freq))

def replace_special_cases(df, final_row, init_row=None):
    def _replace_special_cases(tweet):
        tweet = re.sub(r'instructress', 'instructor', tweet)
        tweet = re.sub(r'nonproduction', 'production', tweet)
        tweet = re.sub(r'sledder', 'sled', tweet)
        tweet = re.sub(r'signless', 'sign', tweet)
        tweet = re.sub(r'wishless', 'wish', tweet)

        return tweet

    if init_row is None:
        init_row = final_row
    
    df[final_row] = df[init_row].apply(_replace_special_cases)

def remove_empty_tweets(df, row):
    df.drop(df[df[row] == ''].index, inplace=True)

def preprocess(df, final_row, init_row):
    remove_links_and_hashtags(df, final_row, init_row=init_row)
    to_lowercase(df, final_row)
    remove_contractions(df, final_row)
    remove_punctuations(df, final_row)
    remove_whitespaces(df, final_row)
    remove_empty_tweets(df, final_row)
    remove_non_english_tweets(df, final_row)
    lemmatize(df, final_row)
    remove_non_english_words(df, final_row)
    remove_english_stopwords(df, final_row)
    remove_common_words(df, final_row)
    remove_rare_words(df, final_row)
    replace_special_cases(df, final_row)
