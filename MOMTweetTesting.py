# MakeoverMonday  Bot Testing

#import libraries
import tweepy
import pandas as pd
import time
import os
os.chdir('C:\\Users\\Sabal\\Desktop\\Data Science Class Files\\Random Practice\\MakeoverMondayProjectScripts')
#from RefinedMOMTweetRetrieverCode import collectTweets
from credentials import *
from tweepy import TweepError

#Keys to initiate connection to twitter api for bot
consumer_key=	consumer_key
consumer_secret=consumer_secret
access_token=access_token
access_token_secret=access_token_secret
auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

#pull tweets
tweetjsons=[]
for tweet in tweepy.Cursor(api.search,q='#makeovermonday',since='2018-04-01', until='2018-04-07',
                           include_entities=True,result_type='recent'
                           ,exlude_replies=True).items():
    if 'RT' not in tweet.text:
        tweetjsons.append(tweet)
        
os.chdir('D:\\MakeoverMondayDataFiles')
realtweets=pd.read_csv('realTweets.csv',encoding='latin1',usecols=['Name','Text'])
     
#retweet remaining tweets
for tweet in tweetjsons:
    try:
        if tweet.author.name in realtweets['Name'].unique().tolist():
            api.retweet(tweet.id)
    except TweepError:
        pass