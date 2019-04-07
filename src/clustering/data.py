from gensim.models import KeyedVectors
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from django.conf import settings

max_data_index = 9600
update_size = 240

w2v = KeyedVectors.load('.{}gensim_w2v.kv'.format(settings.MODEL_URL))

documents = pd.read_csv('.{}scraped_tweets.csv'.format(settings.DATA_URL), dtype=object).dropna().sample(n=max_data_index).reset_index(drop=True)

kmeans = MiniBatchKMeans(n_clusters=8, batch_size=update_size)
pca = None

clustered_docs = {}
clustered_word_count = {}
