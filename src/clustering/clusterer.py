from gensim.models import KeyedVectors
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from django.conf import settings

from collections import Counter
import operator

from .tweet_preprocessor import preprocess, stopwords

class TwitterKMeans:

    def __init__(self, model, n_clusters=8, ttl=3, n_iterations=5):
        self.__model = model

        self.__n_clusters = n_clusters
        self.__ttl = ttl
        self.__n_iterations = n_iterations

        self.__tweets = None
        self.__centroids = None

        self.__pca = None
        self.__scaler = StandardScaler()
    
    """""""""""""""""""""""""""
    Main clustering procedures
    """""""""""""""""""""""""""
    def cluster(self, tweets):
        tweets = preprocess(tweets)

        if self.__tweets is None or self.__centroids is None:
            self.__init_clusters(tweets)
        else:
            self.__increment_clusters(tweets)

        return self.__summarize(self.__centroids)

    def __init_clusters(self, new_tweets):
        self.__tweets = new_tweets
        self.__tweets['ttl'] = self.__ttl

        tweet_vectors = [self.__create_vector(tweet) for tweet in self.__tweets['cleanText'].values]
        
        clustering = KMeans(n_clusters=self.__n_clusters).fit(tweet_vectors) 
        self.__tweets['label'] = clustering.labels_

        self.__centroids = [centroid.tolist() for centroid in clustering.cluster_centers_]
    
    def __increment_clusters(self, new_tweets):
        new_tweets['ttl'] = self.__ttl

        tweet_vectors = [self.__create_vector(tweet) for tweet in new_tweets['cleanText'].values]

        tree = KDTree(self.__centroids)
        active_labels = []
        labels = []
        for vector in tweet_vectors:
            closest_label = self.__centroids.index(self.__centroids[tree.query(vector)[1]])
            if closest_label not in active_labels:
                active_labels.append(closest_label)
            labels.append(closest_label)
        new_tweets['label'] = labels

        inactive_tweets = self.__tweets[~self.__tweets['label'].isin(active_labels)]
        inactive_tweets['ttl'] = inactive_tweets['ttl'] - 1
        self.__tweets = self.__tweets.drop(self.__tweets[self.__tweets['ttl'] <= 0].index)

        self.__tweets = self.__tweets.append(new_tweets)

        new_centroids = []
        for label in range(self.__n_clusters):
            tweets = self.__tweets[self.__tweets['label'] == label]
            centroid = np.mean([self.__create_vector(tweet) for tweet in tweets['cleanText'].values],
                               axis=0).tolist()
            new_centroids.append(centroid)
        
        self.__centroids = new_centroids
    
    """""""""""""""
    Helper methods
    """""""""""""""
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

            tweets = self.__tweets[self.__tweets['label'] == label]

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
    
    def __create_vector(self, tweet):
        word_vectors = [self.__model[word] for word in tweet.split()]

        return np.mean(word_vectors, axis=0).tolist()
