from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Tweets(models.Model):
    tweet_ids = models.IntegerField()

    def __str__(self):
        return str(self.tweet_ids)

