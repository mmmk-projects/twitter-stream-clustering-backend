import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial import KDTree

import nltk
import nltk.corpus as corpus
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import KeyedVectors

from django.conf import settings

from random import choice

class TwitterKMeans:

    def __init__(self, n_clusters=6, ttl=5, iteration=100):
        self.__n_clusters = n_clusters
        self.__ttl = ttl
        self.__iteration = iteration
        
        self.__df = None
        self.__clusters = None
        
        self.__scaler = StandardScaler()
        self.__pca = PCA(n_components=2)
        
        self.__words = set(corpus.words.words())
        self.__stopwords = corpus.stopwords.words('english')
        self.__lemmatizer = WordNetLemmatizer()
        
        self.__w2v = KeyedVectors.load('.{}gensim_w2v.kv'.format(settings.MODEL_URL))
        # self.__w2v = KeyedVectors.load_word2vec_format(
        #     '.{}GoogleNews-vectors-negative300.bin'.format(settings.MODEL_URL),
        #     binary=True)

        self.__tree = None
    
    def cluster(self, df):
        if not self.__is_init():
            self.__initialize(df)
        else:
            self.__cluster(df)
        
        return self.__df, self.__clusters
    
    def __is_init(self) :
        return self.__df and self.__clusters
    
    def __initialize(self, df):
        self.__df = df.copy()
        self.__df['ttl'] = self.__ttl

        tokenized_docs = [self.__preprocess(doc) for doc in self.__df['clean_text'].toList()]
        doc_vectors = [self.__aggregate([self.__w2v[w] for w in doc]) for doc in tokenized_docs]

        self.__tree = KDTree(doc_vectors)
        labels = [0] * len(doc_vectors)
        centroids = self.__init_centroids(doc_vectors)
        for _ in range(self.__iteration):
            for idx, vector in enumerate(doc_vectors):
                max_sim = self.__compute_doc_similarity(vector, centroids[labels[idx]])
                for label, centroid in enumerate(centroids[1:]):
                    sim = self.__compute_doc_similarity(vector, centroid)
                    if sim > max_sim:
                        labels[idx] = label
                        max_sim = sim
            centroids = self.__update_centroids(doc_vectors, labels)
        self.__df['labels'] = labels
        
        self.__clusters = []
        # TODO: Create clusters information

    
    def __cluster(self, df):
        pass
    
    def __preprocess(self, document):
        return [self.__lemmatizer.lemmatize(w) for w in nltk.wordpunct_tokenize(document)
            if w.lower() in self.__words and w.lower() not in self.__stopwords or not w.isalpha()]

    def __aggregate(self, vectors):
        min_vector = []
        max_vector = []

        for i in range(len(vectors)):
            v_list = [v[i] for v in vectors]
            min_v = min(v_list)
            max_v = max(v_list)
            min_vector.append(min_v)
            max_vector.append(max_v)
        
        return min_vector + max_vector

    def __init_centroids(self, vectors):
        centroids = []

        for _ in range(self.__n_clusters):
            centroid = choice(vectors)
            while centroid in centroids:
                centroid = choice(vectors)
            centroids.append(centroid)
        
        return centroids
    
    def __update_centroids(self, vectors, labels):
        centroids = []

        for label in labels:
            centroids[label].append(vectors[self.__tree.query(np.mean(
                [vec for vec, lbl in zip(vectors, labels) if lbl == label], axis=0))[1]])

        return centroids

    def __compute_doc_similarity(self, v1, v2):
        return cosine_similarity(v1, v2)
