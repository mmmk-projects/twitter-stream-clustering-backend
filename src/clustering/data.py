from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
import nltk
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from django.conf import settings

english_words = set(nltk.corpus.words.words())
english_stop_words = nltk.corpus.stopwords.words("english")

def preprocess(text):
    return " ".join(w for w in nltk.wordpunct_tokenize(text)
        if w.lower() in english_words and w.lower() not in english_stop_words or not w.isalpha())

def tokenize(document):
    return simple_preprocess(str(document).encode("utf-8"))

w2v = KeyedVectors.load(".{}gensim_w2v.kv".format(settings.MODEL_URL))

documents = pd.read_csv(".{}train.csv".format(settings.DATA_URL), dtype=object)[["question1"]].dropna().sample(n=3000).reset_index(drop=True)
last_indices = []
words = []
for _, row in documents.iterrows():
    tokens = tokenize(preprocess(row["question1"]))
    if len(last_indices) > 0:
        last_indices.append(last_indices[len(last_indices) - 1] + len(tokens))
    else:
        last_indices.append(len(tokens))
    words.extend(tokens)

vectors = list(map(lambda word: w2v[word], words))

document_vectors = []
first_index = 0
(vector_size,) = w2v[words[0]].shape
for last_index in last_indices:
    vector = []
    for i in range(vector_size):
        v_list = [v[i] for v in vectors[first_index:last_index]]
        if len(v_list) > 0:
            v = sum(v_list) / len(v_list)
            vector.append(v)
    if len(vector) > 0:
        document_vectors.append(vector)
    first_index = last_index
document_vectors = list(document_vectors)

document_vectors = StandardScaler().fit_transform(document_vectors)
document_vectors = PCA(0.85).fit_transform(document_vectors)

document_vectors = TSNE(n_components=2).fit_transform(document_vectors)
