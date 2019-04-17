from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .clusterer import max_data_size
from .data import documents, max_data_index, twitter_kmeans, update_size
from .tweet_preprocessor import preprocess

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
        docs = documents[from_idx:to_idx].copy()
        update_indices()
        while len(docs) == 0:
            docs = documents[from_idx:to_idx].copy()
            update_indices()
        clusters, max_x, max_y, init = twitter_kmeans.cluster(docs)

        return JsonResponse({
            'clusters': clusters,
            'maxX': float(max_x),
            'maxY': float(max_y),
            'isInit': init
        })
    else:
        return HttpResponse(status=405)
