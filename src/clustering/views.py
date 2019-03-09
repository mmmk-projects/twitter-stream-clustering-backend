from gensim.utils import simple_preprocess
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .data import documents, kmeans, max_data_index, pca, update_size, w2v

english_words = set(nltk.corpus.words.words())
english_stop_words = nltk.corpus.stopwords.words('english')

lemmatizer = WordNetLemmatizer()

max_data_size = 240

from_idx, to_idx = 0, 90

def update_indices():
    global from_idx, to_idx

    to_idx += update_size
    if to_idx > max_data_index:
        from_idx = 0
        to_idx = max_data_size
    elif to_idx - from_idx > max_data_size:
        from_idx = to_idx - max_data_size

def preprocess(text):
    return ' '.join(lemmatizer.lemmatize(w) for w, tag in nltk.pos_tag(nltk.wordpunct_tokenize(text))
        if tag[0] == 'N' and w.lower() in english_words and w.lower() not in english_stop_words or not w.isalpha())

def tokenize(document):
    return simple_preprocess(str(document).encode('utf-8'))

@csrf_exempt
def get_clusters(request):
    if request.method == 'GET':
        global pca

        docs = documents[from_idx:to_idx]
        update_indices()
        while len(docs) == 0:
            docs = documents[from_idx:to_idx]
            update_indices()
        
        last_indices = []
        words = []
        for _, row in docs.iterrows():
            tokens = tokenize(preprocess(row['question1']))
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

        clustering = kmeans.partial_fit(document_vectors)
        centers = clustering.cluster_centers_
        centers_reduced = StandardScaler().fit_transform(centers)
        if pca is None:
            pca = PCA(n_components=2)
            centers_reduced = pca.fit_transform(centers_reduced)
        else:
            centers_reduced = pca.transform(centers_reduced)

        labels = list(dict.fromkeys(clustering.labels_))
        label_sizes = {}
        for label in clustering.labels_:
            if label not in label_sizes:
                label_sizes[label] = 0
            label_sizes[label] += 1

        clusters = []
        max_x, max_y = 0, 0
        for center, center_reduced, label in zip(centers, centers_reduced, labels):
            hashtag = w2v.similar_by_vector(np.array(center), topn=1)[0][0]
            x, y = center_reduced[0], center_reduced[1]
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

            clusters.append({
                'id': float(label),
                'hashtag': str(hashtag),
                'x': float(x),
                'y': float(y),
                'size': label_sizes[label]
            })
        clusters.sort(key=lambda c: c['id'])

        return JsonResponse({
            'clusters': clusters,
            'max_x': float(max_x),
            'max_y': float(max_y)
        })
    else:
        return HttpResponse(status=405)
