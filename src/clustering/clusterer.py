from gensim.models import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from django.conf import settings

from collections import Counter
import operator

from .tweet_preprocessor import preprocess, stopwords

class TwitterKMeans:

    def __init__(self, model, n_clusters=6, ttl=5, n_iterations=5):
        self.__model = model

        self.__n_clusters = n_clusters
        self.__ttl = ttl
        self.__n_iterations = n_iterations

        self.__tweets = None

        self.__pca = None
        self.__scaler = StandardScaler()
    
    """""""""""""""""""""""""""
    Main clustering procedures
    """""""""""""""""""""""""""
    def cluster(self, tweets):
        tweets = preprocess(tweets)

        # if self.__tweets is None:
        #     centroids = self.__init_clusters(tweets)
        # else:
        #     centroids = self.__increment_clusters(tweets)
        centroids = self.__init_clusters(tweets)

        return self.__summarize(centroids)

    def __init_clusters(self, tweets):
        self.__tweets = tweets
        self.__tweets['ttl'] = self.__ttl

        tweet_vectors = [self.__create_vector(tweet) for tweet in self.__tweets['cleanText'].values]
        
        clustering = KMeans(n_clusters=self.__n_clusters).fit(tweet_vectors) 
        self.__tweets['labels'] = clustering.labels_

        return [centroid.tolist() for centroid in clustering.cluster_centers_]
    
    def __increment_clusters(self, tweets):
        return []
    
    def __summarize(self, centroids):
        reduced_centroids = self.__scaler.fit_transform(centroids)
        if self.__pca is None:
            self.__pca = PCA(n_components=2)
            reduced_centroids = self.__pca.fit_transform(reduced_centroids)
        else:
            reduced_centroids = self.__pca.transform(reduced_centroids)
        
        clusters = []
        max_x, max_y = 0, 0
        for label in range(self.__n_clusters):
            centroid = centroids[label]
            reduced_centroid = reduced_centroids[label]

            tweets = self.__tweets[self.__tweets['labels'] == label]
            print(tweets)

            documents = [tweet for tweet in tweets[['tweetId', 'time', 'user', 'text', 'cleanText', 'likes', 'retweets', 'replies']].to_dict('records')]
            documents.sort(key=operator.itemgetter('time', 'retweets', 'likes'), reverse=True)

            word_count = Counter([words for doc in documents for words in doc['cleanText'].split()])
            word_count = sorted(word_count.items(), key=lambda wc: wc[1], reverse=True)
            word_count = [{'word': word, 'count': count} for word, count in word_count]
            most_frequent = word_count[0]['word']

            initial_hashtag = self.__model.similar_by_vector(np.array(centroid), topn=1)[0][0]
            cadidate_hashtag = np.mean([self.__model[initial_hashtag], self.__model[most_frequent]],
                                       axis=0)
            idx = 0
            hashtags = self.__model.similar_by_vector(cadidate_hashtag, topn=10)
            hashtag = hashtags[idx][0]
            while hashtag in [cluster['hashtag'] for cluster in clusters]:
                idx += 1
                hashtag = hashtags[idx][0]

            x, y = reduced_centroid[0], reduced_centroid[1]
            if abs(x) > max_x:
                max_x = abs(x)
            if abs(y) > max_y:
                max_y = abs(y)
            
            clusters.append({
                'id': float(label) + 1,
                'hashtag': str(hashtag),
                'x': float(x),
                'y': float(y),
                'size': float(len(documents)),
                'documents': documents,
                'wordCount': word_count
            })
        
        return clusters, max_x, max_y
    
    """""""""""""""
    Helper methods
    """""""""""""""
    def __create_vector(self, tweet):
        word_vectors = [self.__model[word] for word in tweet.split()]

        return np.mean(word_vectors, axis=0).tolist()
