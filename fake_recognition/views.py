import tweepy
import json
import logging
import threading
import time

from django.shortcuts import render
from django.views.generic.base import View
from fake_recognition.forms import DetectFakeForm
from fake_recognition import tweet_feature_generation
from fake_recognition import user_feature_generation
from fake_recognition import retraining
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse


consumer_key= '0jEOVROURtW1KB0XAPwCmJfbG'
consumer_secret= '9mlXK4XCAbnJhF7TOYbNiCvjbhXWG24mAzuqyakY6a00CwXUgj'

access_token='2286177409-bzF4WmAiN23XBEEVoAsoc1VRRRBSJhaLMJoAXSy'
access_token_secret='xsGZqwFIM3nqbsofNf5LxFrWvkORFrND2M2N3KI3Xoyr1'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweet = None

class FakeImageView(View):
    template_name = "index.html"

    def get(self, request):
        #create a form object
        form = DetectFakeForm()
        return render(request, self.template_name, {'form': form})


    def post(self, request):
        tweet_list = []
        if request.method == 'POST':
            form = DetectFakeForm(request.POST)
        if form.is_valid():
            tweetval = form.cleaned_data['tweet_ids']
            tweet_list.append(tweetval)
            form.save()
        tweets = api.statuses_lookup(tweet_list)
        json_tweets = json.dumps(tweets[0]._json)
        tweet = json.loads(json_tweets)

        #tweet_features_classification = tweet_feature_generation.main(tweet)
        #user_features = user_feature_generation.main(tweet)
        retrained_classification = retraining.main(tweet)

        if 'media' in tweet['entities']:
            media_url = tweet['entities']['media'][0]['media_url']
            tweet_text = tweet['text']
        else:
            media_url = 'https://www-10.lotus.com/ldd/portalwiki.nsf/dx/noMedia.jpg/$file/nomedia.jpg'

        return render(request, self.template_name, {'form': form, 'tweet_id':tweetval,'media_url': media_url,
                                                    'tweet_text': tweet_text,
                                                    'retraining':retrained_classification})




"""
class FakeJSONView(View):

    def get(self, request):
        if request.method == 'POST':
            forms = DetectFakeForm(request.POST)
        if forms.is_valid():
            tweetvals = forms.cleaned_data['tweet_ids']
        #id_list = Tweets.objects.all()
        tweetsa = api.statuses_lookup(tweetvals)  # id_list is the list of tweet ids
        return render(request, 'tweet_json.html', {'tweets':tweetsa})
"""


