from gensim.models import KeyedVectors
import pandas as pd

from django.conf import settings

from .clusterer import TwitterKMeans

update_size = 75

documents = pd.read_csv('.{}scraped_tweets.csv'.format(settings.DATA_URL), dtype=object) \
              .dropna(subset=['text']) \
              .reset_index()
documents['cleanText'] = documents['text']
documents.sort_values(by=['time'], inplace=True)

model = KeyedVectors.load('.{}gensim_w2v.kv'.format(settings.MODEL_URL))

pca = None

clustered_docs = {}
clustered_word_count = {}
