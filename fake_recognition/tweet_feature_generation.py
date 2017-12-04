import re
import requests
import urllib, json
import urllib2
import nltk
import csv
import logging
import pandas as pd
import sys
import pickle

from bs4 import BeautifulSoup
from PIL import Image
from textstat.textstat import textstat
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
from urlparse import urlparse

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

F_HARMONIC = "D:/Downloads/hostgraph-h.tsv/hostgraph-h.tsv"
F_INDEGREE = "D:/Downloads/hostgraph-indegree.tsv/hostgraph-indegree.tsv"

F_PLEASE = "C:/Users/imaad/twitteradvancedsearch/senti_words/please.txt"
F_HAPPYEMO = "C:/Users/imaad/twitteradvancedsearch/emoticons/happy-emoticons.txt"
F_SADEMO = "C:/Users/imaad/twitteradvancedsearch/emoticons/sad-emoticons.txt"
F_FIRSTPRON = "C:/Users/imaad/twitteradvancedsearch/pronouns/first-order-prons.txt"
F_SECONDPRON = "C:/Users/imaad/twitteradvancedsearch/pronouns/second-order-prons.txt"
F_THIRDPRON = "C:/Users/imaad/twitteradvancedsearch/pronouns/third-order-prons.txt"
F_SLANG = "C:/Users/imaad/twitteradvancedsearch/slang_words/slangwords.txt"
F_NEGATIVE = "C:/Users/imaad/twitteradvancedsearch/senti_words/negative-words.txt"
F_POSITIVE = "C:/Users/imaad/twitteradvancedsearch/senti_words/positive-words.txt"

def getNumUppercaseChars(tweet):
    count = 0
    postText = tweet['text']
    for i in postText:
        postTextStr = i.encode('utf-8', 'ignore')
    postTextStr =  re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', postTextStr) # URLs
    postTextStr =  re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)','', postTextStr) # remove hash-tags
    postTextStr =  re.sub(r'(?:@[\w_]+)','', postTextStr) # remove @-mentions
    for i in postTextStr:
        if i.isupper():
            count = count+1
    return count


def getNumUrls(tweets_data):
    numUrls1 = [tweet.get('entities''urls','') for tweet in tweets_data]
    numurls1 = []
    if len(numUrls1) > 0:
        #numurls1.extend(tweet.get('entities''media',''))
        numurls1.extend(tweet['entities']['urls'])
        tweet['url1'] = [numurl['url'] for numurl in numurls1]
    if 'media' in tweet['entities']:
        #numUrls2 = map(lambda tweet: tweet['entities']['media'] if tweet['entities'] != None else None, tweets_data)
        numUrls2 = [tweet.get('entities''media','') for tweet in tweets_data]
        numurls2 = []
        if len(numUrls2) > 0:
            numurls2.extend(tweet['entities']['media'])
            tweet['url2'] = [numurl['url'] for numurl in numurls2]
        totalurl = len(tweet['url1']) + len(tweet['url2'])
    else:
        totalurl = len(tweet['url1'])
    return totalurl

def get_alexa_metrics(domain):
    metrics = (0, 0, 0 ,0)
    if domain == None:
        return metrics
    url = "http://data.alexa.com/data?cli=10&url="+ domain
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

def hasExternalLinks(tweets_data):
        extLinks = checkForExternalLinks(tweets_data)
        if extLinks:
            return True
        else:
            return False

def getWotTrustValue(expandUrlName):
	if expandUrlName == None:
		return 0
	parse_obj = urlparse(expandUrlName)
	expandUrlNameStr = str(parse_obj.netloc)
	#if "/" in expandUrlName:
		#expandUrlName1 = expandUrlName.split("/")[2:3]
	#else:
		#expandUrlName1 = expandUrlName.split("/")[:]
	#expandUrlNameStr = str(expandUrlName1[0])
	url = "http://api.mywot.com/0.4/public_link_json2?hosts="+ expandUrlNameStr +"/&key=108d4b2a42ea1afc370e668b39cabdceaa19fcf0"
	#print url
	response = urllib.urlopen(url)
	data = json.load(response)
	#print data
	if data:
		try:
			dataTrust = data[expandUrlNameStr]['0']
			valueTrust = dataTrust[0]
			confTrust = dataTrust[1]
			value = valueTrust * confTrust / 100
			#print value[0]
			return value
		except KeyError:
			return 0

def numNouns(postText):
    is_noun = lambda pos: pos[:2] == 'NN'
    for i in postText:
        postTextStr = i.encode('utf-8', 'ignore')
    #postTextStr = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",postTextStr).split())
    #print postTextStr
    for i in postTextStr:
        tokenized = nltk.word_tokenize(postTextStr)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    countNouns = Counter([j for i,j in pos_tag(word_tokenize(postTextStr))])
    return countNouns['NN']

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
    except Exception,e:
        return False

def expandedUrl(shortenedUrl):
    if shortenedUrl == None:
        return None
    try:
        expandedUrl = (requests.get(shortenedUrl).url)
        expandedUrlName = expandedUrl.split("/")[2:3]
        expandedUrlNameStr = str(expandedUrlName[0])
        expandedUrlNameStr = expandedUrlNameStr.replace("http://","")
        expandedUrlNameStr = expandedUrlNameStr.replace("www.", "")
    except requests.exceptions.ConnectionError as e:
        #print e
        expandedUrlNameStr = None
    return expandedUrlNameStr

def checkForExternalLinks(tweet):
    numurls1 = []
    urlList = []
    numUrls1 = tweet['entities']['urls']
    if len(numUrls1) > 0:
        numurls1.extend(tweet['entities']['urls'])
        tweet['url1'] = [numurl['url'] for numurl in numurls1]
        #print tweet['url1']
        urlList.extend(tweet['url1'])
    #print urlList
    for ural in urlList:
		checkForImage = isAnImage(ural)
		if  not checkForImage:
			return ural
		else:
			return None

def getIndegree(expandedLink, fin=F_INDEGREE):
    #print expandedLink
    if expandedLink == "twitter.com":
        expandedLink = None
    indegreeval = None

    if expandedLink == None:
    	return 0
    with open(fin, 'rb') as tsvin:
    	tsvreader = csv.reader(tsvin,delimiter="\t")
    	for row in tsvreader:
    		if expandedLink == row[0]:
    			return row[1]

def getHarmonic(indegree, expandedLink, f_harmonic):
    if indegree == None:
        return None
    return getIndegree(expandedLink, F_HARMONIC)


def read_pattern(fin, regex=False):
    with open(fin) as f:
        patterns = f.readlines()
    return [re.compile(r'\b%s\b' % p.strip()) for p in patterns]


def pattern_count(ttext, pattern):
    count = 0
    for p in pattern:
        matches = p.findall(ttext)
        count += len(matches)
    return count > 0, count

PATTERN_PLEASE = read_pattern(F_PLEASE)
PATTERN_HAPPYEMO = read_pattern(F_HAPPYEMO)
PATTERN_SADEMO  = read_pattern(F_SADEMO)
PATTERN_FIRSTPRON = read_pattern(F_FIRSTPRON)
PATTERN_SECPRON = read_pattern(F_SECONDPRON)
PATTERN_THIRDPRON = read_pattern(F_THIRDPRON)
PATTERN_SLANG = read_pattern(F_SLANG)
PATTERN_NEGATIVE = read_pattern(F_NEGATIVE)
PATTERN_POSITIVE = read_pattern(F_POSITIVE)

header = ('id',
          'tweetTextLen',
          'numItemWords',
          'questionSymbol',
          'exclamSymbol',
          'numQuesSymbol',
          'numExclamSymbol',
          'happyEmo',
          'sadEmo',
          'numUpperCase',
          'containFirstPron',
          'containSecPron',
          'containThirdPron',
          'numMentions',
          'numHashtags',
          'numUrls',
          'positiveWords',
          'negativeWords',
          'slangWords',
          'pleasePresent',
          'rtCount',
          'colonSymbol',
          'externLinkPresent',
          'Indegree',
          'Harmonic',
          'AlexaPopularity',
          'AlexaReach',
          'AlexaDelta',
          'AlexaCountry',
          'WotValue',
          'numberNouns',
          'readabilityValue')

def gen_features(tweet):
	#tid = tweet['id_str']
	ttext = tweet['text']
	tlength = len(ttext)
	twords = len(ttext.split())
	counts = Counter(ttext)
	questionSymbol = '?' in counts
	exclamSymbol = '!' in counts
	numQuesSymbol = counts.get('?', 0)
	numExclamSymbol = counts.get('!', 0)
	numUpperCase = getNumUppercaseChars(tweet)
	numMentions = len(tweet['entities'].get('user_mentions', []))
	numHashtags = len(tweet['entities'].get('user_hashtags', []))
	numUrls = len(tweet['entities'].get('urls', [])) + len(tweet['entities'].get('media', []))
	rtCount = tweet.get('retweet_count', 0)
	colonSymbol = ':' in counts
	externLinkPresent = len(tweet['entities'].get('urls', [])) > 0
	externalLink = checkForExternalLinks(tweet)
	expandedLink = expandedUrl(externalLink)
	indegree = getIndegree(expandedLink)
	harmonic = getHarmonic(indegree, expandedLink, F_HARMONIC)
	alexa_metrics = get_alexa_metrics(expandedLink)
	wotValue = getWotTrustValue(externalLink)
	numberNouns = numNouns(ttext)
	readabilityValue = textstat.flesch_reading_ease(str(' '.join(ttext).encode('utf-8').strip()))
	please_exists, _ = pattern_count(ttext, PATTERN_PLEASE)
	containsFirstPron, _ = pattern_count(ttext, PATTERN_FIRSTPRON)
	containsSecPron, _ = pattern_count(ttext, PATTERN_SECPRON)
	containsThirdPron, _ = pattern_count(ttext, PATTERN_THIRDPRON)
	containsHappyEmo, _ = pattern_count(ttext, PATTERN_HAPPYEMO)
	containsSadEmo, _ = pattern_count(ttext, PATTERN_SADEMO)
	_, slangWords = pattern_count(ttext, PATTERN_SLANG)
	_, negWords = pattern_count(ttext, PATTERN_NEGATIVE)
	_, posWords = pattern_count(ttext, PATTERN_POSITIVE)
	features = (
                tlength,
                twords,
                questionSymbol,
                exclamSymbol,
				externLinkPresent,
				numberNouns,
				containsHappyEmo,
				containsSadEmo,
				containsFirstPron,
				containsSecPron,
				containsThirdPron,
				numUpperCase,
				posWords,
				negWords,
				numMentions,
                numHashtags,
				numUrls,
				rtCount,
				slangWords,
				colonSymbol,
				please_exists,
				wotValue,
                numQuesSymbol,
                numExclamSymbol,
				readabilityValue,
                indegree,
                harmonic,
                alexa_metrics)

	return features


def linear_reg(item):
    item_new = item.loc[item['AlexaCountry'] != 0]
    item_new_1 = item_new[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol', 'numUpperCase',
                           'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
                           'slangWords', 'rtCount', 'WotValue', 'numberNouns', 'readabilityValue']]
    item_1 = item[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol', 'numUpperCase',
                   'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
                   'slangWords', 'rtCount', 'WotValue', 'numberNouns', 'readabilityValue']]
    lr = LinearRegression()
    X_train = item_new_1
    X_new_vals = item_1
    column_names = ['AlexaCountry', 'AlexaReach', 'AlexaDelta', 'AlexaPopularity', 'Harmonic', 'Indegree']
    Y_train = [item_new['AlexaCountry'], item_new['AlexaReach'], item_new['AlexaDelta'], item_new['AlexaPopularity'],
               item_new['Harmonic'], item_new['Indegree']]
    for i in range(0, len(Y_train)):
        linearmodel = lr.fit(X_train, Y_train[i])
        item[column_names[i]] = lr.predict(X_new_vals)
    return item

def normalize(item):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	item[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
			  'slangWords','rtCount','Indegree','Harmonic', 'AlexaPopularity', 'AlexaReach', 'AlexaDelta',
			  'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']] = scaler.fit_transform(item[['tweetTextLen', 'numItemWords', 'numQuesSymbol', 'numExclamSymbol','numUpperCase', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
			  'slangWords','rtCount','Indegree','Harmonic','AlexaPopularity', 'AlexaReach', 'AlexaDelta',
			  'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']])
	item = MultiColumnLabelEncoder(columns = ['questionSymbol','exclamSymbol','externLinkPresent','happyEmo', 'sadEmo', 'containFirstPron',
											'containSecPron', 'containThirdPron','colonSymbol','pleasePresent' ]).fit_transform(item_df)
	return item

class MultiColumnLabelEncoder:
	def __init__(self,columns = None):
		self.columns = columns # array of column names to encode

	def fit(self,X,y=None):
		return self # not relevant here

	def transform(self,X):
		'''
		Transforms columns of X specified in self.columns using
		LabelEncoder(). If no columns specified, transforms all
		columns in X.
		'''
		output = X.copy()
		if self.columns is not None:
			for col in self.columns:
				output[col] = LabelEncoder().fit_transform(output[col])
		else:
			for colname,col in output.iteritems():
				output[colname] = LabelEncoder().fit_transform(col)
		return output

	def fit_transform(self,X,y=None):
		return self.fit(X,y).transform(X)

def preprocess(df):
    df = linear_reg(df)
    df = normalize(df)
    return df

def main(tweet, tweet_features=None, tweet_predictions=None):
    features = gen_features(tweet)
    #return features

    flatten = lambda lst: reduce(lambda l, i: l + flatten(i) if isinstance(i, (list, tuple)) else l + [i], lst, [])
    features = flatten(features)
    df = pd.DataFrame([features],
                      columns=['tweetTextLen', 'numItemWords', 'questionSymbol', 'exclamSymbol', 'externLinkPresent',
                               'numberNouns', 'happyEmo', 'sadEmo', 'containFirstPron', 'containSecPron',
                               'containThirdPron',
                               'numUpperCase', 'positiveWords', 'negativeWords', 'numMentions', 'numHashtags',
                               'numUrls',
                               'rtCount', 'slangWords', 'colonSymbol', 'pleasePresent', 'WotValue', 'numQuesSymbol',
                               'numExclamSymbol', 'readabilityValue', 'Indegree', 'Harmonic',
                               'AlexaPopularity', 'AlexaReach', 'AlexaCountry', 'AlexaDelta'])

    with open('fake_recognition/models/tweet_models.pkl', 'rb') as f:
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
    return df, pred_val[0]
