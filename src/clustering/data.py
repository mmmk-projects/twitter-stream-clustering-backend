from gensim.models import KeyedVectors
import pandas as pd

from django.conf import settings

from .clusterer import TwitterKMeans

max_data_index = 9600
update_size = 240

documents = pd.read_csv('.{}scraped_tweets.csv'.format(settings.DATA_URL), dtype=object) \
              .dropna(subset=['text']) \
              .reset_index()
word_counts = pd.Series(' '.join(documents['text']).split()).value_counts()
common_freq = word_counts[word_counts > 300]
rare_freq = word_counts[word_counts <= 3]
documents['cleanText'] = documents['text'].apply(lambda text: ' '.join(word for word in text.split()
                                                                       if word not in common_freq
                                                                       and word not in rare_freq))

model = KeyedVectors.load_word2vec_format('.{}glove.6B.100d.word2vec'.format(settings.MODEL_URL))

twitter_kmeans = TwitterKMeans(model)
pca = None

clustered_docs = {}
clustered_word_count = {}
