from gensim.models import KeyedVectors
import pandas as pd

from django.conf import settings

from .clusterer import TwitterKMeans

max_data_index = 9600
update_size = 240

documents = pd.read_csv('.{}scraped_tweets.csv'.format(settings.DATA_URL), dtype=object) \
              .dropna(subset=['text']) \
              .reset_index()
word_count = pd.Series(' '.join(documents['text']).split()).value_counts()
rare_freq = word_count[word_count <= 3]
documents['cleanText'] = documents['text'].apply(lambda text: ' '.join(word for word in text.split()
                                                                       if word not in rare_freq))
documents.sort_values(by=['time'], inplace=True)

model = KeyedVectors.load_word2vec_format('.{}glove.twitter.27B.100d.word2vec'.format(settings.MODEL_URL))

twitter_kmeans = TwitterKMeans(model)
pca = None

clustered_docs = {}
clustered_word_count = {}
