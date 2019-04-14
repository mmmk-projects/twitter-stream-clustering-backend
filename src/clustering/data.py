from gensim.models import KeyedVectors
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from django.conf import settings

max_data_index = 9600
update_size = 240

w2v = KeyedVectors.load_word2vec_format('.{}glove.6B.100d.word2vec'.format(settings.MODEL_URL))

documents = pd.read_csv('.{}scraped_tweets.csv'.format(settings.DATA_URL), dtype=object) \
              .dropna(subset=['text']) \
              .sample(frac=1) \
              .reset_index()

kmeans = MiniBatchKMeans(n_clusters=8, batch_size=update_size)
pca = None

clustered_docs = {}
clustered_word_count = {}
