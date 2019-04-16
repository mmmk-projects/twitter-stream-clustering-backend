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

def remove_links_and_hashtags(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: tweet.clean(str(text)))

    return df

def to_lowercase(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: text.lower())

    return df

def remove_contractions(df):
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
    
    df['cleanText'] = df['cleanText'].apply(_remove_contractions)

    return df

def remove_punctuations(df):
    df['cleanText'] = df['cleanText'].str.replace(r'[^\w\s]', '')

    return df

def remove_whitespaces(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: str(text).strip())

    return df

def remove_non_english_tweets(df):
    df['lang'] = df['cleanText'].apply(lambda text: detect(text))
    df = df.drop(df[df['lang'] != 'en'].index)
    df = df.drop(columns=['lang'])

    return df

def lemmatize(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: ' '.join(lemmatizer.lemmatize(word)
                                                                  for word in text.split()))

    return df

def remove_non_english_words(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: ' '.join(word for word in text.split()
                                                                  if word in words
                                                                  and len(word) > 1))

    return df

def remove_english_stopwords(df):
    df['cleanText'] = df['cleanText'].apply(lambda text: ' '.join(word for word in text.split()
                                                                  if word not in stopwords))

    return df

def replace_special_cases(df):
    def _replace_special_cases(tweet):
        tweet = re.sub(r'instructress', 'instructor', tweet)
        tweet = re.sub(r'nonproduction', 'production', tweet)
        tweet = re.sub(r'sledder', 'sled', tweet)
        tweet = re.sub(r'signless', 'sign', tweet)
        tweet = re.sub(r'wishless', 'wish', tweet)

        return tweet

    df['cleanText'] = df['cleanText'].apply(_replace_special_cases)

    return df

def remove_empty_tweets(df):
    df = df.drop(df[df['cleanText'] == ''].index)

    return df

def preprocess(df):
    df = remove_links_and_hashtags(df)
    df = to_lowercase(df)
    df = remove_contractions(df)
    df = remove_punctuations(df)
    df = remove_whitespaces(df)
    df = remove_empty_tweets(df)
    df = remove_non_english_tweets(df)
    df = lemmatize(df)
    df = remove_non_english_words(df)
    df = remove_english_stopwords(df)
    df = remove_empty_tweets(df)
    df = replace_special_cases(df)

    return df
