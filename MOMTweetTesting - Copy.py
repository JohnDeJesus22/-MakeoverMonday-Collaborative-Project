# MakeoverMonday 

#import libraries
import tweepy
import pandas as pd
import time
from oauth2client.service_account import ServiceAccountCredentials

#Keys to initiate connection to twitter api for bot
consumer_key=	''
consumer_secret=''
access_token=''
access_token_secret=''

auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

rob=api.get_user('@robcrock')

test='Test tweet @johnnydata22'
api.update_status(test)

tweetjsons=[]
for tweet in tweepy.Cursor(api.user_timeline,id='@robcrock',since='2018-03-17', until='2018-03-21',
                           include_entities=True,result_type='recent'
                           ,exlude_replies=True).items():
    if 'RT' not in tweet.text:
        tweetjsons.append(tweet)

tweetjsons[0].retweet()