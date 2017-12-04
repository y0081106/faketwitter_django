from django.conf.urls import url
from views import FakeImageView
from . import views

urlpatterns = [
    url(r'^', FakeImageView.as_view(),name='index.html'),
    #url(r'^tweet_json/', FakeJSONView.as_view(),name='tweet_json.html')
]