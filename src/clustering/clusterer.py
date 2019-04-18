from gensim.models import KeyedVectors
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabaz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from django.conf import settings

from collections import Counter
import math
import operator

from .tweet_preprocessor import preprocess, stopwords

max_data_size = 118

class TwitterKMeans:

    def __init__(self, model, n_clusters=8, fading=0.9, thresh=0.1, n_iterations=5):
        self.__model = model

        self.__n_clusters = n_clusters
        self.__fading = fading
        self.__thresh = thresh
        self.__n_iterations = n_iterations
        self.__is_init = False

        self.__tweets = None
        self.__centroids = None

        self.__pca = None
        self.__scaler = StandardScaler()
    
    """""""""""""""""""""""""""
    Main clustering procedures
    """""""""""""""""""""""""""
    def cluster(self, tweets):
        self.__init = False

        tweets = preprocess(tweets)

        if self.__tweets is None or self.__centroids is None:
            self.__init_clusters(tweets)
        else:
            self.__increment_clusters(tweets)

        """""""""
        Evaluation
        """""""""
        # active = self.__tweets['ttl'] > self.__thresh
        # X = [self.__create_vector(tweet) for tweet in self.__tweets[active]['cleanText'].values]
        # labels = self.__tweets[active]['label'].values
        # print('Sillhouette score:', silhouette_score(X, labels))
        # print('Calinzki-Harabaz score:', calinski_harabaz_score(X, labels))
        # print('Davies-Bouldin score:', davies_bouldin_score(X, labels))

        return self.__summarize(self.__centroids)

    def __init_clusters(self, new_tweets, init=True):
        self.__is_init = True

        self.__tweets = new_tweets
        if init:
            self.__tweets['ttl'] = 1

        tweet_vectors = [self.__create_vector(tweet) for tweet in
                         self.__tweets[self.__tweets['ttl'] > self.__thresh]['cleanText'].values]
        
        clustering = KMeans(n_clusters=self.__n_clusters).fit(tweet_vectors) 
        self.__tweets['label'] = clustering.labels_

        self.__centroids = [centroid.tolist() for centroid in clustering.cluster_centers_]
    
    def __increment_clusters(self, new_tweets):
        self.__is_init = False

        new_tweets['ttl'] = 1

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

        is_active = self.__tweets['label'].isin(active_labels)
        self.__tweets.loc[is_active, 'ttl'] = 1
        self.__tweets.loc[~is_active, 'ttl'] = self.__tweets['ttl'] * self.__fading

        self.__tweets = self.__tweets.append(new_tweets)

        new_centroids = []
        for label in range(self.__n_clusters):
            tweets = self.__tweets[(self.__tweets['label'] == label)
                                   & (self.__tweets['ttl'] > self.__thresh)]
            if len(tweets.index) > 0:
                vectors = [self.__create_vector(tweet) for tweet in tweets['cleanText'].values]
                centroid = np.mean(vectors, axis=0).tolist()

                new_centroids.append(centroid)
        if len(new_centroids) < self.__n_clusters:
            self.__init_clusters(self.__tweets, init=False)
            return

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
        max_size = max_data_size
        max_x, max_y = 0, 0
        for label in range(self.__n_clusters):
            centroid = centroids[label]
            reduced_centroid = reduced_centroids[label]

            tweets = self.__tweets[(self.__tweets['label'] == label)
                                   & (self.__tweets['ttl'] > self.__thresh)]

            cols = ['tweetId', 'time', 'user', 'text', 'cleanText', 'likes', 'retweets', 'replies']
            documents = [tweet for tweet in tweets[cols].to_dict('records')]
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
            while  hashtag in stopwords or hashtag in [cluster['hashtag'] for cluster in clusters]:
                idx += 1
                hashtag = hashtags[idx][0]

            x, y = reduced_centroid[0], reduced_centroid[1]
            if abs(x) > max_x:
                max_x = abs(x)
            if abs(y) > max_y:
                max_y = abs(y)
            
            size = len(documents)
            if size > max_size:
                max_size = size
            
            clusters.append({
                'id': float(label) + 1,
                'hashtag': str(hashtag),
                'x': float(x),
                'y': float(y),
                'size': float(size),
                'documents': documents,
                'wordCount': word_count
            })
        if max_size > max_data_size:
            scale_ratio = math.ceil(max_size / (max_data_size * 2.0 / 3.0))
            for cluster in clusters:
                cluster['size'] = math.ceil(cluster['size'] / scale_ratio)
        
        return clusters, max_x, max_y, self.__is_init
    
    def __create_vector(self, tweet):
        word_vectors = [self.__model[word] for word in tweet.split()]

        return np.mean(word_vectors, axis=0).tolist()
