from django.forms import ModelForm
from .models import Tweets

class DetectFakeForm(ModelForm):
    class Meta:
        model = Tweets
        fields = ['tweet_ids',]

