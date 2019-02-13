import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .data import document_vectors, kmeans, max_data_index, max_data_size, pca, update_size, w2v

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

        documents = document_vectors[from_idx:to_idx]
        update_indices()
        while len(documents) == 0:
            documents = document_vectors[from_idx:to_idx]
            update_indices()

        clustering = kmeans.partial_fit(documents)
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
