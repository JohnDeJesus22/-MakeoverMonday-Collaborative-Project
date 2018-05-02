# ML Model Version 2

import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords #to use stopwords function and list
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize.moses import MosesDetokenizer #had to nltk download 'perluniprops' first

#change directory and get data
os.chdir('D:\\MakeoverMondayDataFiles')
falsetweets=pd.read_csv('NewFalsetweets.csv',encoding='latin1')
realtweets=pd.read_csv('NewRealTweets.csv',encoding='latin1')

#create labels for classification
realtweets['RealorFake']=1
falsetweets['RealorFake']=0

#combine real and false tweets, drop lat and long, and set index
model_data=pd.concat([realtweets,falsetweets],axis=0)
model_data=model_data.set_index([[i for i in range(model_data.shape[0])]])

#shuffle the data
from sklearn.utils import shuffle
model_data=shuffle(model_data,random_state=0)
###############################################################################################
#setting up tweets for train/test split
#initialize corpus
corpus=[]

#create instances of tweettokenizer, detokenizer, and porterstemmer.
#initialize pattern to filter urls and usernames
twtoken=TweetTokenizer()
detokenizer=MosesDetokenizer()
ps=PorterStemmer()
url_pattern=re.compile(r'https\S+')
user_pattern=re.compile(r'@\S+')

#build corpus
for i in range(model_data.shape[0]):
    text=model_data['Text'][i]
    urls=re.findall(url_pattern,text)
    users=re.findall(user_pattern,text)
    users=[re.sub('[^@a-zA-z]','',user) for user in users]
    text=twtoken.tokenize(text)
    for url in urls:
        if url in text:
            text.remove(url)
    for user in users:
        if user in text:
            text.remove(user)
    text=detokenizer.detokenize(text,return_str=True)
    text=re.sub('[^a-zA-z]',' ',text)
    text=text.lower()
    text=text.split()
    try:
        text.remove('makeovermonday')
    except Exception:
        pass
    text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text=' '.join(text)
    corpus.append(text)
    
###############################################################################################
#testing with bag of words model with tfidftransformer and spliting data
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000,strip_accents='ascii')# returns 1000 columns of most frequent words
X=cv.fit_transform(corpus).toarray()
y=model_data.loc[:,'RealorFake'].values

from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer()
X=tfidf.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test=train_test_split(X,y,
                                                test_size=0.20, random_state=0)

#############################################################################################
#Fitting MultiNomial Naive Bayes to Training set (should have used this first since
#it is mainly for text classification)

#4/22/18: Improved results with removing 'makeovermonday' from tweets, tfidfvectorizer, and
#below version of naive bayes. Average accuracy about 55% with some accuracies 
#reaching 60%-64% and 75%
#4/25/18: More data and some parameter tuning helped! Got mean accuracy of 77% and a variance
#of .01! I just hope that the false tweets were represented enough...
#roc_auc is 0.5424 and may still need more false tweets since there are a large # of 
#false positives.
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB(alpha=.87,fit_prior=True)#setting fit-prior=false improved variance
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuray=accuracy_score(y_test,y_pred)

#k-fold cross validation 
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=X_train,y=y_train,n_jobs=1,cv=10)
mean=accuracies.mean() #check accuracy(bias)
variance=accuracies.std()#determine variance

#import libraries
import tweepy
import pandas as pd
import time
import os
os.chdir('C:\\Users\\Sabal\\Desktop\\Data Science Class Files\\Random Practice\\MakeoverMondayProjectScripts\\Twitter_Bot')
#from RefinedMOMTweetRetrieverCode import collectTweets
from credentials import *
from tweepy import TweepError
from RefinedMOMTweetRetrieverCode import TweetCollector as TC

#Keys to initiate connection to twitter api for bot
consumer_key=	consumer_key
consumer_secret=consumer_secret
access_token=access_token
access_token_secret=access_token_secret
auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)


tweetjsons=[]
for tweet in tweepy.Cursor(api.search,
                           q="#makeovermonday ",since='2018-04-17', until='2018-05-02',
                           include_entities=True,result_type='recent'
                           ,exlude_replies=True).items():
    if 'RT' not in tweet.text:#remove retweets
        tweetjsons.append(tweet)

viz_info=[]
for tweet in tweetjsons:
    ids=tweet.id
    name=tweet.author.name
    screen_name=tweet.author.screen_name
    try:
        location=tweet.author.location
    except Exception:
        location=None
    try:
        time_zone=tweet.author.time_zone
    except Exception:
        time_zone=None
    language=tweet.lang
    date=tweet.created_at
    try:
        tweet_url=tweet.entities['urls'][0]['url']
    except Exception:
        tweet_url='NoLink'
        text=tweet.text
        viz_info.append((ids,name,screen_name,date,tweet_url,text,location,time_zone,language))
            
df=pd.DataFrame(viz_info, columns=['ID','Name','TwitterHandle', 'Date', 'TweetUrl','Text','Location',
                                           'TimeZone','Language'])

prediction_corpus=[]

for i in range(df.shape[0]):
    text=df['Text'][i]
    urls=re.findall(url_pattern,text)
    users=re.findall(user_pattern,text)
    users=[re.sub('[^@a-zA-z]','',user) for user in users]
    text=twtoken.tokenize(text)
    for url in urls:
        if url in text:
            text.remove(url)
    for user in users:
        if user in text:
            text.remove(user)
    text=detokenizer.detokenize(text,return_str=True)
    text=re.sub('[^a-zA-z]',' ',text)
    text=text.lower()
    text=text.split()
    try:
        text.remove('makeovermonday')
    except Exception:
        pass
    text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text=' '.join(text)
    prediction_corpus.append(text)


New_tweets=cv.transform(prediction_corpus).toarray()
X_nt=tfidf.transform(New_tweets) 

predictions=classifier.predict(X_nt) 

df['Results']=predictions

for i in df['ID']:
    try:
        if tweet.author.name in realtweets['ID']:
            api.retweet(tweet.id)
    except TweepError:
        pass

#determining Roc_auc score
from sklearn import metrics
y_pred_prob = classifier.predict_proba(X_test)[:, 1]
y_pred_prob_percentage=sum(y_pred_prob>.5)/len(y_pred_prob)
roc_auc=metrics.roc_auc_score(y_test, y_pred_prob)
