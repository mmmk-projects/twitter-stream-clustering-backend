import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from collections import Counter
import operator

from .data import clustered_docs, clustered_word_count, documents, \
                  kmeans, max_data_index, pca, update_size, w2v
from .tweet_preprocessor import preprocess

max_data_size = 118

from_idx, to_idx = 0, 90

def update_indices():
    global from_idx, to_idx

    to_idx += update_size
    if to_idx > max_data_index:
        from_idx = 0
        to_idx = max_data_size
    elif to_idx - from_idx > max_data_size:
        from_idx = to_idx - max_data_size

@csrf_exempt
def get_clusters(request):
    if request.method == 'GET':
        global pca

        docs = documents[from_idx:to_idx].copy()
        update_indices()
        while len(docs) == 0:
            docs = documents[from_idx:to_idx]
            update_indices()
        preprocess(docs, 'clean_text', 'text')
        
        last_indices = []
        words = []
        for _, row in docs.iterrows():
            tokens = row['clean_text'].split()
            if len(last_indices) > 0:
                last_indices.append(last_indices[len(last_indices) - 1] + len(tokens))
            else:
                last_indices.append(len(tokens))
            words.extend(tokens)

        vectors = list(map(lambda word: w2v[word], words))

        document_centers = []
        document_vectors = []
        first_index = 0
        (vector_size,) = w2v[words[0]].shape
        for last_index in last_indices:
            vector = []
            min_vector = []
            max_vector = []
            for i in range(vector_size):
                v_list = [v[i] for v in vectors[first_index:last_index]]
                if len(v_list) > 0:
                    v = sum(v_list) / len(v_list)
                    min_v = min(v_list)
                    max_v = max(v_list)
                    vector.append(v)
                    min_vector.append(min_v)
                    max_vector.append(max_v)
            if len(vector) > 0:
                document_centers.append(vector)
                min_vector.extend(max_vector)
                document_vectors.append(min_vector)
            first_index = last_index
        document_centers = list(document_centers)
        document_vectors = list(document_vectors)

        clustering = kmeans.partial_fit(document_vectors)
        labels = clustering.labels_
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
        for center, center_reduced, label in zip(document_centers, centers_reduced, labels):
            member_docs = []
            for idx, row in enumerate(docs.to_dict('records')):
                if idx < len(clustering.labels_) and clustering.labels_[idx] == label:
                    member_docs.append(row)
            if label not in clustered_docs:
                clustered_docs[label] = []
            clustered_docs[label].extend(member_docs)
            clustered_docs_size = len(clustered_docs[label])
            if clustered_docs_size > max_data_size:
                clustered_docs[label] = clustered_docs[label][clustered_docs_size-max_data_size:]
            clustered_docs[label].sort(key=operator.itemgetter('time', 'retweets', 'likes'), reverse=True)

            doc_words = []
            for doc in clustered_docs[label]:
                doc_words.extend(doc['clean_text'].split())
            word_count = Counter(doc_words)
            sorted_word_count = sorted(word_count.items(), key=lambda kv: kv[1], reverse=True)
            clustered_word_count[label] = [{'word': word, 'count': count} for word, count in sorted_word_count]

            hashtag = w2v.similar_by_vector(np.array(center), topn=1)[0][0]
            x, y = center_reduced[0], center_reduced[1]
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

            clusters.append({
                'id': float(label) + 1,
                'hashtag': str(hashtag),
                'x': float(x),
                'y': float(y),
                'size': label_sizes[label],
                'documents': clustered_docs[label],
                'word_count': clustered_word_count[label]
            })
        clusters.sort(key=lambda c: c['id'])

        return JsonResponse({
            'clusters': clusters,
            'max_x': float(max_x),
            'max_y': float(max_y)
        })
    else:
        return HttpResponse(status=405)
