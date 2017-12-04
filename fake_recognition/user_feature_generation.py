import numpy as np
import pandas as pd
import datetime as dt
import urllib, json
import json
import re
import requests
import string
import urllib2
import time
import nltk
import csv
import pickle
import emoji
import sys

from PIL import Image
from urlparse import urlparse
from datetime import datetime
from nltk.tag import pos_tag
from langdetect import detect
from bs4 import BeautifulSoup
from collections import Counter
from dateutil.parser import parse
from geopy.geocoders import Nominatim
from textstat.textstat import textstat
from nltk.tokenize import word_tokenize

F_HARMONIC = "D:/Downloads/hostgraph-h.tsv/hostgraph-h.tsv"
F_INDEGREE = "D:/Downloads/hostgraph-indegree.tsv/hostgraph-indegree.tsv"


def hasExistingLocation(tweets):
    try:
        geolocator = Nominatim()
        existingLocation = geolocator.geocode(getLocation(tweets))
        if existingLocation == None:
            return False
        if (existingLocation.address):
            return True
        else:
            return False
    except:
        return False


def get_alexa_metrics(domain):
    metrics = (0, 0, 0, 0)
    if domain == None:
        return metrics
    url = "http://data.alexa.com/data?cli=10&url=" + domain
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        popularity = soup.popularity['text']
        rank = soup.reach['rank']
        country = soup.country['rank']
        delta = soup.rank['delta']
        return (popularity, rank, country, delta)
    except TypeError:
        return metrics


def isAnImage(url):
    text = 'pbs.twimg.com'
    text1 = 'p.twimg.com'
    if re.search(url, text) or re.search(url, text1):
        return True
    try:
        im = Image.open(urllib2.urlopen(url))
        if im != None:
            return True
        else:
            return False
    except Exception, e:
        return False


def checkForExternalLinks(tweet):
    numurls1 = []
    urlList = []
    numUrls1 = tweet['entities']['urls']
    if len(numUrls1) > 0:
        numurls1.extend(tweet['entities']['urls'])
        tweet['url1'] = [numurl['url'] for numurl in numurls1]
        # print tweet['url1']
        urlList.extend(tweet['url1'])
    # print urlList
    for ural in urlList:
        checkForImage = isAnImage(ural)
        if not checkForImage:
            return ural
        else:
            return None


def hasExternalLinks(tweets_data):
    extLinks = checkForExternalLinks(tweets_data)
    if extLinks:
        return True
    else:
        return False


def expandedUrl(shortenedUrl):
    if shortenedUrl == None:
        return None
    expandedUrl = (requests.get(shortenedUrl).url)
    expandedUrlName = expandedUrl.split("/")[2:3]
    expandedUrlNameStr = str(expandedUrlName[0])
    expandedUrlNameStr = expandedUrlNameStr.replace("http://", "")
    expandedUrlNameStr = expandedUrlNameStr.replace("www.", "")
    return expandedUrlNameStr


def expandUrl(shortenedUrl):
    expandedUrl = (requests.get(shortenedUrl).url)
    expandedUrlName = expandedUrl.split("/")[2:3]
    expandedUrlNameStr = str(expandedUrlName[0])
    return expandedUrlNameStr


def getWotTrustValue(expandUrlName):
    if expandUrlName == None:
        return 0
    parse_obj = urlparse(expandUrlName)
    expandUrlNameStr = str(parse_obj.netloc)
    # if "/" in expandUrlName:
    # expandUrlName1 = expandUrlName.split("/")[2:3]
    # else:
    # expandUrlName1 = expandUrlName.split("/")[:]
    # expandUrlNameStr = str(expandUrlName1[0])
    url = "http://api.mywot.com/0.4/public_link_json2?hosts=" + expandUrlNameStr + "/&key=108d4b2a42ea1afc370e668b39cabdceaa19fcf0"
    # print url
    response = urllib.urlopen(url)
    data = json.load(response)
    # print data
    if data:
        try:
            dataTrust = data[expandUrlNameStr]['0']
            valueTrust = dataTrust[0]
            confTrust = dataTrust[1]
            value = valueTrust * confTrust / 100
            # print value[0]
            return value
        except KeyError:
            return 0


def numMediaContent(tweet):
    userName = tweet['user']['screen_name'] if tweet['user'] != None else None
    for i in userName:
        userNameStr = str(i).encode('utf-8', 'ignore')
    url = "https://twitter.com/" + userNameStr
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        for e in soup.findAll('div', {'class': 'PhotoRail'}):
            photoid = e.find('span', {'class': 'PhotoRail-headingText'}).text.strip()
            photoids = str(photoid)
        numMedia = [float(s) for s in photoid.split() if
                    all(j.isdigit() or j in string.punctuation for j in s) and any(j.isdigit() for j in s)]
        for i in numMedia:
            numMediaValue = i
        return i
    except:
        return 0


def hasProfileImg(tweets_data):
    profileImg = map(lambda tweet: tweet['user']['default_profile_image'], tweets_data)
    for i in profileImg:
        profileImgStr = str(i).encode('utf-8', 'ignore')
    return profileImgStr


def hasHeaderImg(tweets_data):
    headerImg = map(lambda tweet: tweet['user']['profile_use_background_image'], tweets_data)
    # headerImgStr = ''.join(str(i) for i in headerImg)
    for i in headerImg:
        headerImgStr = str(i).encode('utf-8', 'ignore')
    return headerImgStr


def getAccountAge(tweet):
    age = tweet['user']['created_at']
    for i in age:
        ageStr = i.encode('utf-8', 'ignore')
    age1 = parse(ageStr)
    age1 = str(age1)
    try:
        mytime = datetime.strptime(age1, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')
    except ValueError as v:
        if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
            age1 = age1[:-(len(v.args[0]) - 26)]
            mytime = datetime.strptime(age1, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')
    # start of epoch time
    epoch = dt.datetime.utcfromtimestamp(0)
    # plugin your time object
    my_time = dt.datetime.strptime(mytime, "%Y-%m-%d %H:%M:%S")
    delta = my_time - epoch
    return delta.total_seconds()


def getTweetRatio(tweet, accountAge, numTweets):
    timeCreated = tweet['created_at']
    for i in timeCreated:
        timeCreatedStr = i.encode('utf-8', 'ignore')
    # timeCreatedStr = ''.join(str(i) for i in timeCreated)
    # print timeCreatedStr
    timeCreatedStr1 = parse(timeCreatedStr)
    timeCreatedStr1 = str(timeCreatedStr1)

    try:
        mytime = datetime.strptime(timeCreatedStr1, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')
    except ValueError as v:
        if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
            timeCreatedStr1 = timeCreatedStr1[:-(len(v.args[0]) - 26)]
            mytime = datetime.strptime(timeCreatedStr1, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')

    # start of epoch time
    epoch = dt.datetime.utcfromtimestamp(0)
    # plugin your time object
    my_time = dt.datetime.strptime(mytime, "%Y-%m-%d %H:%M:%S")
    delta = my_time - epoch
    timeCreatedValue = delta.total_seconds()

    if accountAge != None:
        try:
            tweetRatio = (float(numTweets) / float(timeCreatedValue - accountAge) * 86400L)
            tweetRatio = str(tweetRatio)
            return tweetRatio
        except:
            return 0
    else:
        return 0


def getIndegree(expandedLink, fin=F_INDEGREE):
    # print expandedLink
    if expandedLink == "twitter.com":
        expandedLink = None
    indegreeval = None

    if expandedLink == None:
        return 0
    with open(fin, 'rb') as tsvin:
        tsvreader = csv.reader(tsvin, delimiter="\t")
        for row in tsvreader:
            if expandedLink == row[0]:
                return row[1]


def getHarmonic(indegree, expandedLink, f_harmonic):
    if indegree == None:
        return None
    return getIndegree(expandedLink, F_HARMONIC)


def gen_features(tweet):
    tid = tweet['id_str']
    numFriends = tweet['user'].get('friends_count', 0)
    numFollowers = tweet['user'].get('followers_count', 0)
    if numFollowers != 0:
        folFriendRatio = float(numFriends) / (numFollowers)
    else:
        folFriendRatio = 0
    timesListed = tweet['user']['listed_count']
    hasUrl = len(tweet['user'].get('urls', [])) > 0
    isVerified = tweet['user'].get('verified', [])
    numTweets = tweet['user'].get('statuses_count', 0)
    hasBio = len(tweet['user'].get('description', [])) > 0
    hasLocation = len(tweet['user'].get('location', [])) > 0
    hasExistingLoc = hasExistingLocation(tweet)
    externalLink = checkForExternalLinks(tweet)
    expandedLink = expandedUrl(externalLink)
    wotValue = getWotTrustValue(externalLink)
    numMedia = numMediaContent(tweet)
    accountAge = getAccountAge(tweet)
    hasProfileImg = tweet['user']['default_profile_image']
    hasHeaderImg = tweet['user']['profile_use_background_image']
    tweetRatio = getTweetRatio(tweet, accountAge, numTweets)
    indegree = getIndegree(expandedLink)
    harmonic = getHarmonic(indegree, expandedLink, F_HARMONIC)
    alexa_metrics = get_alexa_metrics(expandedLink)

    features = (
                numFriends,
                numFollowers,
                folFriendRatio,
                timesListed,
                hasUrl,
                isVerified,
                numTweets,
                hasBio,
                hasLocation,
                hasExistingLoc,
                wotValue,
                numMedia,
                accountAge,
                hasProfileImg,
                hasHeaderImg,
                tweetRatio,
                indegree,
                harmonic,
                alexa_metrics)

    return features

def main(tweet, user_predictions=None):
    features = gen_features(tweet)
    flatten = lambda lst: reduce(lambda l, i: l + flatten(i) if isinstance(i, (list, tuple)) else l + [i], lst, [])
    features = flatten(features)
    df = pd.DataFrame([features])

    with open('fake_recognition/models/user_models.pkl', 'rb') as f:
        tweet_model_list = pickle.load(f)
    preds = []
    for model in tweet_model_list:
        preds.append(model.predict(df))

    pred_val = []
    result = Counter(preds[i][0] for i in range(len(preds)))
    res_key_val = result.keys(), result.values()
    len(res_key_val)
    if len(res_key_val) >= 2:
        if res_key_val[1][0] > res_key_val[1][1]:
            pred_val.append("real")
        else:
            pred_val.append("fake")
    else:
        pred_val.append(res_key_val[0][0])
    return pred_val[0]