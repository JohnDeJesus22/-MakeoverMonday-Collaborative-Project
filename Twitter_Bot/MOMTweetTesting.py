# MakeoverMonday  Bot Testing

'''
Outline of script
1. Collect tweets
2.Run ML model to filter tweets
3.Retweet TP and FN tweets
'''

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



collected_tweets= TC.collectandStore('2018-04-26','2018-04-30')
collected_tweets=TC.GeocodeDF(collected_tweets)


#retweet remaining tweets
for tweet in tweetjsons:
    try:
        if tweet.author.name in realtweets['ID']:
            api.retweet(tweet.id)
    except TweepError:
        pass