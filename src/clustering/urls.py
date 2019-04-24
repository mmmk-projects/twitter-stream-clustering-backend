from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.get_clusters),
    url(r'^reset/$', views.reset)
]
