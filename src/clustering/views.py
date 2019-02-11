import numpy as np
from sklearn.cluster import KMeans

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .data import document_vectors, document_vectors_reduced, max_data_index, max_data_size, update_size, w2v

from_idx, to_idx = 0, 100

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
        documents = document_vectors[from_idx:to_idx]
        documents_reduced = document_vectors_reduced[from_idx:to_idx]
        update_indices()

        clustering = KMeans(5).fit(documents)
        labels = clustering.labels_
        clusters_dict = {}
        clusters_dict_reduced = {}
        for doc_coordinate, doc_coordinate_reduced, label in zip(documents, documents_reduced, labels):
            if label not in clusters_dict and label not in clusters_dict_reduced:
                clusters_dict[label] = []
                clusters_dict_reduced[label] = []
            clusters_dict[label].append(doc_coordinate)
            clusters_dict_reduced[label].append(doc_coordinate_reduced)
        
        clusters = []
        max_x, max_y = 0, 0
        for label in clusters_dict_reduced.keys():
            size = len(clusters_dict_reduced[label])
            x_list = [doc_coordinate[0] for doc_coordinate in clusters_dict_reduced[label]]
            y_list = [doc_coordinate[1] for doc_coordinate in clusters_dict_reduced[label]]
            x = sum(x_list) / len(x_list)
            y = sum(y_list) / len(y_list)
            if abs(x) > max_x:
                max_x = abs(x)
            if abs(y) > max_y:
                max_y = abs(y)

            vector = []
            for i in range(len(clusters_dict[label][0])):
                axis_list = [doc_coordinate[i] for doc_coordinate in clusters_dict[label]]
                vector.append(sum(axis_list) / len(axis_list))
            hashtag, _ = w2v.similar_by_vector(np.array(vector), topn=1)[0]
            
            clusters.append({
                'hashtag': str(hashtag),
                'x': float(x),
                'y': float(y),
                'size': int(size)
            })
        clusters.sort(key=lambda c: c['hashtag'])
        
        return JsonResponse({
            'clusters': clusters,
            'max_x': float(max_x),
            'max_y': float(max_y)
        })
    else:
        return HttpResponse(status=405)
