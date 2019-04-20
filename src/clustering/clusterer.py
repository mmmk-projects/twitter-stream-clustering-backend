from gensim.models import KeyedVectors
from nltk.tag import pos_tag
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabaz_score, davies_bouldin_score, \
                            silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from django.conf import settings

from collections import Counter
import math
import operator
import time

from .tweet_preprocessor import preprocess, stopwords

max_data_size = 250
max_cluster_size = 125

class TwitterKMeans:

    def __init__(self, model, n_clusters=5, fading=0.85, thresh=0.25):
        self.__model = model

        self.__n_clusters = n_clusters
        self.__fading = fading
        self.__thresh = thresh
        self.__is_init = False

        self.__tweets = None

        self.__clusterer = KMeans(n_clusters=self.__n_clusters)
        self.__centroids = None

        self.__pca = None
        self.__scaler = StandardScaler()

        self.__calinski_harabaz_score = None
        self.__davies_bouldin_score = None
        self.__silhouette_score = None
        self.__silhouette_scores = None
    
    """""""""""""""""""""""""""
    Main clustering procedures
    """""""""""""""""""""""""""
    def cluster(self, tweets):
        start_time = time.time()

        self.__init = False

        tweets = preprocess(tweets)

        if self.__tweets is None or self.__centroids is None:
            self.__init_clusters(tweets)
        else:
            self.__increment_clusters(tweets)
        split_or_merge = False
        self.__evalute_clusters()

        """""""""
        Evaluation
        """""""""
        print()
        print('Clustering took {:.3f} seconds'.format(time.time() - start_time))
        if split_or_merge:
            self.__evalute_clusters()
        print('Calinzki-Harabaz score: {:.3f}'.format(self.__calinski_harabaz_score))
        print('Davies-Bouldin score: {:.3f}'.format(self.__davies_bouldin_score))
        print('Silhouette score: {:.3f}'.format(self.__silhouette_score))
        print('Silhouette score per cluster:')
        for label, score in enumerate(self.__silhouette_scores):
            print('Cluster {}: {:.3f}'.format(label + 1, score))

        return self.__summarize(self.__centroids)

    def __init_clusters(self, new_tweets, init=True):
        self.__is_init = True

        self.__tweets = new_tweets
        if init:
            self.__tweets['ttl'] = 1

        tweet_vectors = [self.__create_vector(tweet) for tweet in
                         self.__tweets[self.__tweets['ttl'] > self.__thresh]['cleanText'].values]
        
        clustering = self.__clusterer.fit(tweet_vectors) 
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
        for label in range(len(self.__centroids)):
            tweets = self.__tweets[(self.__tweets['label'] == label)
                                   & (self.__tweets['ttl'] > self.__thresh)]
            if len(tweets.index) > 0:
                vectors = [self.__create_vector(tweet) for tweet in tweets['cleanText'].values]
                centroid = np.mean(vectors, axis=0).tolist()

                new_centroids.append(centroid)
        if len(new_centroids) < len(self.__centroids):
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
        max_x, max_y = 0, 0
        for label in range(len(self.__centroids)):
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

            initial_hashtag = [hashtag[0] for hashtag in
                               self.__model.similar_by_vector(np.array(centroid), topn=1)][0]
            cadidate_vector = np.mean([self.__model[initial_hashtag], self.__model[most_frequent]],
                                      axis=0)
            idx = 0
            hashtags = [hashtag[0] for hashtag in
                        self.__model.similar_by_vector(cadidate_vector, topn=25)]
            noun_hashtags = [hashtag[0] for hashtag in pos_tag(hashtags) if hashtag[1][0] == 'N']
            try:
                hashtag = noun_hashtags[idx]
                while hashtag in stopwords or hashtag in [cluster['hashtag'] for cluster in clusters]:
                    idx += 1
                    hashtag = noun_hashtags[idx]
            except IndexError:
                hashtag = hashtags[0]

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
        
        return clusters, max_x, max_y, self.__is_init
    
    def __create_vector(self, tweet):
        def __w2v(word):
            try:
                return self.__model[word]
            except KeyError:
                return [0.0] * self.__model.vector_size

        word_vectors = [__w2v(word) for word in tweet.split()]

        return np.mean(word_vectors, axis=0).tolist()
    
    def __evalute_clusters(self):
        active = self.__tweets['ttl'] > self.__thresh
        X = [self.__create_vector(tweet) for tweet in self.__tweets[active]['cleanText'].values]
        labels = self.__tweets[active]['label'].values
        self.__calinski_harabaz_score = calinski_harabaz_score(X, labels)
        self.__davies_bouldin_score = davies_bouldin_score(X, labels)
        self.__silhouette_score = silhouette_score(X, labels)
        silhouette_scores = silhouette_samples(X, labels)
        self.__silhouette_scores = []
        for label in range(len(self.__centroids)):
            score = np.mean([score for idx, score in enumerate(silhouette_scores)
                             if label == labels[idx]])
            self.__silhouette_scores.append(score)
    