import json
from sklearn.cluster import KMeans

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .data import document_words, document_vectors

@csrf_exempt
def get_clusters(request):
    if request.method == 'POST':
        request_body = json.loads(request.body.decode('utf-8'))
        documents = document_vectors[request_body['from_idx']:request_body['to_idx']]

        clustering = KMeans().fit(documents)
        labels = clustering.labels_
        clusters_dict = {}
        for doc_coordinate, label in zip(documents, labels):
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
            words = []
            for d, l in zip(document_words[request_body['from_idx']:request_body['to_idx']], labels):
                if l == label:
                    words.extend(d)
            hashtag = max(set(words), key=words.count)
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
