from gensim.models import KeyedVectors
import pandas as pd

from django.conf import settings

from .clusterer import TwitterKMeans

max_data_index = 9600
update_size = 240

documents = pd.read_csv('.{}scraped_tweets.csv'.format(settings.DATA_URL), dtype=object) \
              .dropna(subset=['text']) \
              .reset_index()
documents['cleanText'] = documents['text']
documents.sort_values(by=['time'], inplace=True)

model = KeyedVectors.load_word2vec_format('.{}glove.6B.100d.word2vec'.format(settings.MODEL_URL))

twitter_kmeans = TwitterKMeans(model)
pca = None

clustered_docs = {}
clustered_word_count = {}
