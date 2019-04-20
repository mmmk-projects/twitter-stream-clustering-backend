from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .clusterer import max_data_size
from .data import documents, twitter_kmeans, update_size
from .tweet_preprocessor import preprocess

from_idx, to_idx = 0, update_size

def update_indices():
    global from_idx, to_idx

    to_idx += update_size
    if to_idx > len(documents.index):
        from_idx = 0
        to_idx = max_data_size
    else:
        from_idx += update_size

@csrf_exempt
def get_clusters(request):
    if request.method == 'GET':
        docs = documents.iloc[from_idx:to_idx].copy()
        update_indices()
        while len(docs.index) == 0:
            docs = documents.iloc[from_idx:to_idx].copy()
            update_indices()

        clusters, max_x, max_y = twitter_kmeans.cluster(docs)

        return JsonResponse({
            'clusters': clusters,
            'maxX': float(max_x),
            'maxY': float(max_y)
        })
    else:
        return HttpResponse(status=405)
