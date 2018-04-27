# MakeoverMonday  Bot Testing

#import libraries
import tweepy
import pandas as pd
import time
import os
os.chdir('C:\\Users\\Sabal\\Desktop\\Data Science Class Files\\Random Practice\\MakeoverMondayProjectScripts\\Twitter_Bot')
#from RefinedMOMTweetRetrieverCode import collectTweets
from credentials import *
from tweepy import TweepError
from RefinedMOMTweetRetrieverCode import collectTweets, GeocodeDF

#Keys to initiate connection to twitter api for bot
consumer_key=	consumer_key
consumer_secret=consumer_secret
access_token=access_token
access_token_secret=access_token_secret
auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

collected_tweets= collectTweets('2018-04-26','2018-04-27')
collected_tweets['ID']=collected_tweets['ID'].astype('str')
collected_tweets=GeocodeDF(collected_tweets)

#pull tweets
tweetjsons=[]
for tweet in tweepy.Cursor(api.search,q='#makeovermonday',since='2018-04-26', until='2018-04-28',
                           include_entities=True,result_type='recent'
                           ,exlude_replies=True).items():
    if 'RT' not in tweet.text:
        tweetjsons.append(tweet)


#retweet remaining tweets
for tweet in tweetjsons:
    try:
        if tweet.author.name in realtweets['Name'].unique().tolist():
            api.retweet(tweet.id)
    except TweepError:
        pass