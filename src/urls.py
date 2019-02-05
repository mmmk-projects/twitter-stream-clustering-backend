from django.conf.urls import include, url

urlpatterns = [
    url(r'^clustering/', include('src.clustering.urls'))
]
