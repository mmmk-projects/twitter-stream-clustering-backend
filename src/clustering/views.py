import json
from sklearn.cluster import KMeans

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .data import document_vectors

@csrf_exempt
def get_clusters(request):
    if request.method == 'POST':
        request_body = json.loads(request.body.decode('utf-8'))
        documents = document_vectors[request_body['from_idx']:request_body['to_idx']]

        clustering = KMeans().fit(documents)
        clusters_dict = {}
        for doc_coordinate, label in zip(documents, clustering.labels_):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(doc_coordinate)
        
        clusters = []
        max_x, max_y = 0, 0
        for label in clusters_dict.keys():
            size = len(clusters_dict[label])
            x_list = [doc_coordinate[0] for doc_coordinate in clusters_dict[label]]
            y_list = [doc_coordinate[1] for doc_coordinate in clusters_dict[label]]
            x = sum(x_list) / len(x_list)
            y = sum(y_list) / len(y_list)
            if abs(x) > max_x:
                max_x = abs(x)
            if abs(y) > max_y:
                max_y = abs(y)
            clusters.append({
                'hashtag': int(label + 1),
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
